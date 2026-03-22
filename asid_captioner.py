import torch
import numpy as np
import gc
from PIL import Image, ImageOps
from math import ceil
import comfy.model_management as mm
import folder_paths
import os
import datetime
import hashlib
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig, set_seed
from .vision_process import process_vision_info, to_rgb
from .audio_process import process_audio_info
# from .vision_process import (
    # extract_vision_info,
    # fetch_image,
    # fetch_video,
    # process_vision_info,
    # smart_resize,
# )
import imageio_ffmpeg



def process_mm_info(
    conversations,
    use_audio_in_video,
    return_video_kwargs=False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
):
    fake_ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if not os.path.exists(fake_ffmpeg_path):
        im_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        os.link(im_ffmpeg_path, fake_ffmpeg_path)
    os.environ["PATH"] += os.pathsep + os.path.dirname(fake_ffmpeg_path)
    
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(
        conversations,
        return_video_kwargs=return_video_kwargs,
        return_video_metadata=return_video_metadata,
        image_patch_size=image_patch_size,
    )
    return (audios,) + vision



# --- 1. Qwen 模型缓存 ---
# 用于存储加载的模型，避免每次节点执行时都重新加载（时间效率优化）
QWEN_MODEL_CACHE = {}
_model_cache_size = 2
QWEN_RESULT_CACHE = {}
_result_cache_size = 20
_batch_cache_size = 100


# --- 2. Qwen 模型加载函数 (使用指定的 Qwen2_5OmniForConditionalGeneration) ---
def load_qwen_components(model_dir: str, dtype: str):
    """加载 Qwen 模型、处理器和分词器，支持 4bit 量化。"""

    if dtype == "4bit":
        # 显存优化：使用 4bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif dtype == "8bit":
        # 显存优化：使用 8bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            #bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif dtype == "fp8":
        # 显存优化：使用 fp8 量化
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float8_e4m3fn,
            device_map="auto"
        )
    else:
        # 完整精度或 Auto 精度加载
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto"
        )

    # 处理器和分词器
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    #tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model.eval()
    model.disable_talker()
    return model, processor


# --- 3. 图像处理函数 ---
def resize_to_limit(image: Image.Image, max_side: int):
    """强制将图像最大边长缩放到指定限制，并确保是 Qwen2.5 所需的 28 的倍数。"""
    width, height = image.size
    
    # 仅当超过限制时才进行缩放
    if max(width, height) > max_side:
        # 1. 计算缩放比例
        ratio = max_side / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
        
    # 2. 确保尺寸是 28 的倍数（Qwen2.5 要求）
    IMAGE_FACTOR = 28
    new_width = ceil(width / IMAGE_FACTOR) * IMAGE_FACTOR
    new_height = ceil(height / IMAGE_FACTOR) * IMAGE_FACTOR

    return image.resize((int(new_width), int(new_height)), resample=Image.BICUBIC)

def get_image_hash(pil_img: Image.Image):
    if pil_img is None:
        return "none"

    # 1. 修复 EXIF 旋转（手机照片常见），确保像素排列顺序与 ComfyUI LoadImage 节点一致
    # 2. 强制转换为 RGB（移除 Alpha 通道干扰）
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    
    # 3. 转换为 uint8 numpy 数组
    # 使用 numpy 数组作为中间体，确保内存布局是 C-style 连续的
    img_np = np.array(img).astype(np.uint8)
    
    # 4. 生成 MD5
    return hashlib.md5(img_np.tobytes()).hexdigest()

# --- 4. 支持从文件读取提示词 ---
def load_prompt_from_file(file_path: str, lang: str):
    """从文件读取对应语言的多行提示词，默认返回预设提示词"""
    # 默认提示词（多行用三引号包裹）
    default_en = "Provide a comprehensive description of all the content in the video, leaving out no details, and naturally covering the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a single coherent account."
    default_zh = "请对视频中的所有内容进行全面描述，不得遗漏任何细节，并将场景、人物、物体、动作、叙事元素、言语、镜头表现以及情感自然地融入一段连贯的叙述中。"
    
    # 文件不存在则返回默认值
    if not os.path.exists(file_path):
        #print(file_path)
        return default_en if lang == "English" else default_zh
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]  # 预处理每行（去首尾空格）
        
        current_lang = None
        prompt_lines = []
        
        for line in lines:
            # 跳过空行和注释
            if not line or line.startswith('#') or line.startswith('/'):
                continue
            
            # 检测语言标识行（如 ==中文==）
            if line.startswith('==') and line.endswith('=='):
                current_lang = line.strip('=').strip()  # 提取语言名称（如 "中文"）
                if current_lang == lang:
                    prompt_lines = []  # 重置当前语言的提示词列表
                continue
            
            # 收集当前语言的提示词行（仅当语言匹配时）
            if current_lang == lang:
                prompt_lines.append(line)
        
        # 如果收集到有效提示词，拼接为多行文本（保留换行）
        if prompt_lines:
            return '\n'.join(prompt_lines)
        
        # 未找到对应语言的提示词，返回默认值
        return default_en if lang == "English" else default_zh
    
    except Exception as e:
        #print(f"read prompt file fail: {e}")
        return default_en if lang == "English" else default_zh


# ----------------------------------------------------------------------------------
# --- 4. ComfyUI 节点类 ---
# ----------------------------------------------------------------------------------

class ASID_Caption:
    def __init__(self):
        # 初始化实例变量，用于存储模型组件
        self.model = None
        self.processor = None
        #self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"image": ("IMAGE", ), # ComfyUI 的图像输入 Tensor
                "model_path": (folder_paths.get_filename_list("text_encoders"), ),
                "dtype": (["auto", "4bit", "8bit"], {"default": "auto"}), # 强烈建议默认 4bit
                "keep_model_loaded": ("BOOLEAN", {"default": False}), # 默认保持加载
                "unload_other_models": ("BOOLEAN", {"default": True}), # 默认卸载其它模型
                "lang": (["中文", "English"], {"default": "中文"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffff}),
                "video_path": ("STRING", {"multiline": False}),
                "use_audio": ("BOOLEAN", {"default": True}), # 是否使用视频中的音频，无音频时选True可能会报错
                "max_side": ("INT", {"default": 504, "min": 252, "max": 2240, "step": 28}), # 默认安全尺寸
                #"instruction": ("STRING", {"multiline": True}),
            },
            "optional": {
                # "image": ("IMAGE", ), # ComfyUI 的图像输入 Tensor
                # "audio": ("AUDIO", ), # ComfyUI 的音频输入
                # "video_fps": ("FLOAT", {"default": 16, "min": 1, "max": 200, "step": 0.1}), # 视频帧率
                "instruction": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "caption"
    CATEGORY = "image/caption"
    OUTPUT_NODE = True


    def caption(self, model_path: str, lang: str, dtype: str, max_side: int, keep_model_loaded: bool, unload_other_models: bool, seed: int, video_path: str, use_audio: bool, instruction: str = None):
        
        if unload_other_models:
            mm.cleanup_models_gc()
            mm.unload_all_models()
        
        set_seed(seed)
        
        VIDEO_MAX_PIXELS = 401408        # 512*28*28
        VIDEO_TOTAL_PIXELS = 20070400    # 512*28*28*50
        USE_AUDIO_IN_VIDEO = use_audio
        
        # --- C. 构造消息和提示词模板 ---
        #if lang == "English":
        #     text_prompt = "Describe this image in detail. Use English"
            #"You are an expert AI Art prompt engineer. Your task is to analyze the image and directly generate one single, detailed, high-quality English prompt optimized for any text-to-image AI model. DO NOT ask any questions or engage in conversation; **strictly output the prompt itself, and nothing else**. Ensure the output is a single, comma-separated string covering style, lighting, subject, and quality tags."
            #"You are an expert AI Art prompt engineer. Based on the input image, generate a single, detailed, and creative high-quality English prompt optimized for any text-to-image AI model."
        #else:
        #     text_prompt = "详细描述这张图片，使用中文"
            #"你是一名专业的AI绘画提示词工程师。你的任务是：根据输入的图像，直接且详细地生成一条高品质、可用于文生图模型的中文提示词。**不要提问或进行任何形式的对话，直接输出结果，只输出提示词本身。** 请确保提示词包含风格、光影、主体和高质量标签。"
            #"你是一名专业的AI绘画提示词工程师。请根据输入的图像，生成详细、富有创意且可以直接用于文生图模型的高品质中文提示词。"
        prompts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
        if not instruction or not instruction.strip():
            text_prompt = load_prompt_from_file(prompts_file, lang)  # 自动返回多行文本
        else:
            if lang == "中文":
                text_prompt = instruction + "，使用中文"
            elif lang == "English":
                text_prompt = instruction + ". Use English"
            elif lang == "bbox":
                text_prompt = instruction + "，返回它们的最小边界框坐标列表。结果必须是一个Python列表的列表，即 [[x1, y1, x2, y2], [x3, y3, x4, y4], ...] 格式。坐标要求：所有坐标值必须是整数。坐标是归一化的，范围是0到1000（表示 0% 到 100% 乘以 10）。每个边界框的顺序为：[左上角X, 左上角Y, 右下角X, 右下角Y]。示例输出：[[250, 150, 450, 500], [600, 700, 800, 950]]请仅输出这个列表结构，不包含任何解释性文字或代码块。"
        print(text_prompt)
        
        model_dir = os.path.dirname(folder_paths.get_full_path_or_raise("text_encoders", model_path))
        
        result_key = (model_dir, dtype, text_prompt+str(seed), video_path+str(max_side))
        
        if result_key in QWEN_RESULT_CACHE:
            self.model, self.processor = None, None
            # --- 显存清理 ---
            if not keep_model_loaded:
                QWEN_MODEL_CACHE.clear()
                mm.cleanup_models_gc()
            gc.collect()
            mm.soft_empty_cache()
            output_text = QWEN_RESULT_CACHE[result_key]
            print(output_text)
            return {"ui": {"text": (output_text,)}, "result": (output_text,)}
        
        # --- A. 模型加载/复用 (时间效率优化) ---
        model_key = (model_dir, dtype)
        if model_key not in QWEN_MODEL_CACHE:
            #print(f"Qwen2.5 VL: 首次加载模型 {model_dir}...")
            try:
                self.model, self.processor = load_qwen_components(model_dir, dtype)
            except Exception as e:
                self.model, self.processor = None, None
                return {"ui": {"text": ("Failed to load model, 模型加载失败",)}, "result": ("Failed to load model, 模型加载失败",)} 
            QWEN_MODEL_CACHE[model_key] = (self.model, self.processor)
            # Limit cache size to prevent OOM
            if len(QWEN_MODEL_CACHE) > _model_cache_size:
                # Remove oldest entries
                keys_to_remove = list(QWEN_MODEL_CACHE.keys())[:len(QWEN_MODEL_CACHE) - _model_cache_size]
                for key in keys_to_remove:
                    del QWEN_MODEL_CACHE[key]
        else:
            self.model, self.processor = QWEN_MODEL_CACHE[model_key]
        
        conversation = [
            # {
                # "role": "system",
                # "content": [
                    # {
                        # "type": "text",
                        # "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    # }
                # ],
            # },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": VIDEO_MAX_PIXELS},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(self.model.device)
        #inputs = inputs.to("cuda")
        
        #self.model.disable_talker()
        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                do_sample=False,
                max_new_tokens=4096,
                repetition_penalty=1.1,
                #use_cache=True,
                return_audio=False,
            )
        
        # 解码并清理
        decoded = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0] # 取第一个解码结果
        output_text = decoded.split("\nassistant\n")[-1].strip()
        print(output_text)
        
        # Cache the result
        QWEN_RESULT_CACHE[result_key] = output_text

        # Limit cache size to prevent memory growth
        if len(QWEN_RESULT_CACHE) > _result_cache_size:
            # Remove oldest entries
            keys_to_remove = list(QWEN_RESULT_CACHE.keys())[:len(QWEN_RESULT_CACHE) - _result_cache_size]
            for key in keys_to_remove:
                del QWEN_RESULT_CACHE[key]
        
        self.model, self.processor = None, None
        # --- 显存清理 ---
        if not keep_model_loaded:
            QWEN_MODEL_CACHE.clear()
            mm.cleanup_models_gc()     
        # 强制清理 GPU 缓存 (显存优化)
        gc.collect()
        mm.soft_empty_cache()
        
        return {"ui": {"text": (output_text,)}, "result": (output_text,)}