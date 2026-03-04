import torch
import numpy as np
import gc
from PIL import Image, ImageOps
from math import ceil
# 必须使用用户指定的类，并确保 BitsAndBytesConfig 导入
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig 
# 假设 vision_process 位于同一目录或可导入
from .vision_process import process_vision_info, to_rgb
import comfy.model_management as mm
import folder_paths
import os
import datetime
import hashlib
#import re


# --- 1. Qwen 模型缓存 ---
# 用于存储加载的模型，避免每次节点执行时都重新加载（时间效率优化）
QWEN_MODEL_CACHE = {}
_model_cache_size = 2
QWEN_RESULT_CACHE = {}
_result_cache_size = 20
_batch_cache_size = 100


# --- 2. Qwen 模型加载函数 (使用指定的 Qwen3_5ForConditionalGeneration) ---
def load_qwen_components(model_dir: str, dtype: str):
    """加载 Qwen 模型、处理器和分词器，支持 4bit 量化。"""

    if dtype == "4bit":
        # 显存优化：使用 4bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
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
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif dtype == "fp8":
        # 显存优化：使用 fp8 量化
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float8_e4m3fn,
            device_map="auto"
        )
    else:
        # 完整精度或 Auto 精度加载
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto"#transformers5.2这里效果有变化
        )

    # 处理器和分词器
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model.eval()
    return model, processor


# --- 3. 图像处理函数 ---
def resize_to_limit(image: Image.Image, max_side: int):
    """强制将图像最大边长缩放到指定限制，并确保是 Qwen 所需的 32 的倍数。"""
    width, height = image.size
    
    # 仅当超过限制时才进行缩放
    if max(width, height) > max_side:
        # 1. 计算缩放比例
        ratio = max_side / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
        
    # 2. 确保尺寸是 32 的倍数（Qwen3 要求）
    IMAGE_FACTOR = 32
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
    default_zh = "详细描述这张图片，使用中文"
    default_en = "Describe this image in detail. Use English"
    
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

def remove_think_text(text: str) -> str:
    """
    通过查找最后一个 </think> 标签，获取其后的正式回复。
    """
    # rpartition 返回一个三元组: (分隔符前的部分, 分隔符, 分隔符后的部分)
    _, separator, final_text = text.rpartition('</think>')
    
    # 如果没找到 </think>，separator 会为空，此时返回原文本
    if not separator:
        return text.strip()
    
    return final_text.strip()


# ----------------------------------------------------------------------------------
# --- 4. ComfyUI 节点类 ---
# ----------------------------------------------------------------------------------

class Qwen35Caption:
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
                "thinking": ("BOOLEAN", {"default": False}), # enable_thinking
                "lang": (["中文", "English", "bbox"], {"default": "中文"}),
                "max_side": ("INT", {"default": 512, "min": 256, "max": 2240, "step": 32}), # 默认安全尺寸
                #"instruction": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image": ("IMAGE", ), # ComfyUI 的图像输入 Tensor
                "video_fps": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}), # 视频帧率
                "instruction": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "full_output")
    FUNCTION = "caption"
    CATEGORY = "image/caption"
    OUTPUT_NODE = True


    def caption(self, model_path: str, lang: str, dtype: str, max_side: int, keep_model_loaded: bool, unload_other_models: bool, thinking: bool, instruction: str = None, video_fps = 16, image: torch.Tensor = None):
        
        if unload_other_models:
            mm.cleanup_models_gc()
            mm.unload_all_models()

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
        
        # --- B. 图像预处理 ---
        
        if image is None:#无图
            #return {"ui": {"text": ("no image, 无图像",)}, "result": ("no image, 无图像", "no image, 无图像")} 
            result_key = (model_dir, dtype, text_prompt+str(thinking), "none")
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
                return {"ui": {"text": (output_text,)}, "result": (remove_think_text(output_text), output_text)}
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": text_prompt },
                    ],
                }
            ]
        elif image.ndim == 4 and image.shape[0] > 1:#多图
            #pil_image_1 = Image.fromarray((image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
            pil_image_mid = Image.fromarray((image[image.shape[0]//2].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
            result_key = (model_dir, dtype, text_prompt+str(thinking), get_image_hash(pil_image_mid)+str(image.shape[0])+str(video_fps)+str(max_side))
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
                return {"ui": {"text": (output_text,)}, "result": (remove_think_text(output_text), output_text)}
            
            video_frames_processed = []
            for i in range(image.shape[0]):
                # 逐帧转换 (N, H, W, C) -> (H, W, C)
                pil_frame = Image.fromarray((image[i].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
                # 处理alpha
                # if pil_frame.mode == 'RGBA':
                # # 调用 vision_process.py 里的逻辑转为 RGB
                    # pil_frame = to_rgb(pil_frame) 
                # elif pil_frame.mode != 'RGB':
                    # pil_frame = pil_frame.convert('RGB')
                pil_frame_resized = resize_to_limit(pil_frame, max_side)
                video_frames_processed.append(pil_frame_resized)
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "video", "video": video_frames_processed, "fps": video_fps }, # 使用缩放后的 PIL Image
                        { "type": "text", "text": text_prompt },
                    ],
                }
            ]
        else:#单图
            # 处理维度
            if image.ndim == 4:
                image_tensor = image.squeeze(0) # 移除 Batch 维度，形状变为 (H, W, C)
            else:
                #what's this?
                image_tensor = image
            # 转换
            pil_image = Image.fromarray((image_tensor.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
            
            result_key = (model_dir, dtype, text_prompt+str(thinking), get_image_hash(pil_image)+str(max_side))
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
                return {"ui": {"text": (output_text,)}, "result": (remove_think_text(output_text), output_text)}
            
            # 处理alpha
            # if pil_image.mode == 'RGBA':
                # # 调用 vision_process.py 里的逻辑转为 RGB
                # pil_image = to_rgb(pil_image) 
            # elif pil_image.mode != 'RGB':
                # pil_image = pil_image.convert("RGB")
            # 2. 强制预缩放
            pil_image_resized = resize_to_limit(pil_image, max_side)
            # 构造消息和提示词模板
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "image", "image": pil_image_resized }, # 使用缩放后的 PIL Image
                        { "type": "text", "text": text_prompt },
                    ],
                }
            ]
        
        # --- A. 模型加载/复用 (时间效率优化) ---
        model_key = (model_dir, dtype)
        if model_key not in QWEN_MODEL_CACHE:
            #print(f"Qwen2.5 VL: 首次加载模型 {model_dir}...")
            try:
                self.model, self.processor = load_qwen_components(model_dir, dtype)
            except Exception as e:
                self.model, self.processor = None, None
                return {"ui": {"text": ("Failed to load model, 模型加载失败",)}, "result": ("Failed to load model, 模型加载失败", "Failed to load model, 模型加载失败")} 
            QWEN_MODEL_CACHE[model_key] = (self.model, self.processor)
            # Limit cache size to prevent OOM
            if len(QWEN_MODEL_CACHE) > _model_cache_size:
                # Remove oldest entries
                keys_to_remove = list(QWEN_MODEL_CACHE.keys())[:len(QWEN_MODEL_CACHE) - _model_cache_size]
                for key in keys_to_remove:
                    del QWEN_MODEL_CACHE[key]
        else:
            self.model, self.processor = QWEN_MODEL_CACHE[model_key]
        
        # --- D. 预处理、推理和解码 ---
        # inputs = self.processor.apply_chat_template(
            # messages,
            # tokenize=True,
            # add_generation_prompt=True,
            # return_dict=True,
            # return_tensors="pt"
        # )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        # 依赖外部 vision_process 导入的函数
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs
            )
        else:   
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                do_resize=False,
                return_tensors="pt"
            )
        # inputs = self.processor(
            # text=[text],
            # images=image_inputs,
            # videos=video_inputs,
            # padding=True,
            # return_tensors="pt",
        # )
        inputs = inputs.to(self.model.device)
        #inputs = inputs.to("cuda")
        if thinking:
            max_token=4096
        else:
            max_token=1024
        
        QWEN35_PRESETS = {
            "VL_No_Think": {
                "temperature": 0.7, "top_p": 0.80, "top_k": 20, 
                "min_p": 0.0, "repetition_penalty": 1.0#, "presence_penalty": 1.5
            },
            "VL_Think": {
                "temperature": 0.6, "top_p": 0.95, "top_k": 20, 
                "min_p": 0.0, "repetition_penalty": 1.0#, "presence_penalty": 0.0
            },
            "Text_No_Think": {
                "temperature": 1.0, "top_p": 1.00, "top_k": 20, 
                "min_p": 0.0, "repetition_penalty": 1.0#, "presence_penalty": 2.0
            },
            "Text_Think": {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, 
                "min_p": 0.0, "repetition_penalty": 1.0#, "presence_penalty": 1.5
            }
        }
        preset_key = ""
        if image is not None:
            if thinking:
                preset_key = "VL_Think"
            else:
                preset_key = "VL_No_Think"
        else:
            if thinking:
                preset_key = "Text_Think"
            else:
                preset_key = "Text_No_Think"
        params = QWEN35_PRESETS[preset_key]
        # 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_token,
                **params
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0] # 取第一个解码结果
        
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
        
        return {"ui": {"text": (output_text,)}, "result": (remove_think_text(output_text), output_text)}
        

class Qwen35CaptionBatch:
    def __init__(self):
        # 初始化实例变量，用于存储模型组件
        self.model = None
        self.processor = None
        #self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (folder_paths.get_filename_list("text_encoders"), ),
                "lang": (["中文", "English"], {"default": "中文"}),
                "dtype": (["auto", "4bit", "8bit"], {"default": "4bit"}), # 强烈建议默认 4bit
                "keep_model_loaded": ("BOOLEAN", {"default": False}), # 默认保持加载
                "max_side": ("INT", {"default": 512, "min": 256, "max": 2240, "step": 32}), # 默认安全尺寸
                "image_path": ("STRING", {"default": ""}),
                },
            "optional": {
                "save_path": ("STRING", {"default": ""}),
                "instruction": ("STRING", {"multiline": True}),
                }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "batch_caption"
    CATEGORY = "image/caption"
    OUTPUT_NODE = True


    def batch_caption(self, model_path: str, lang: str, dtype: str, max_side: int, keep_model_loaded: bool, image_path: str, instruction: str, save_path: str = ""):
        
        count = 0
            
        # 1. 验证输入路径
        if not image_path or not os.path.isdir(image_path):
            error = "0 image captioned, 共处理0张图片"
            return {"ui": {"text": (error,)}, "result": (error,)}
        
        # 2. 确定保存路径（为空则使用image_path）
        if not save_path:
            save_path = image_path
        os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在
		
        model_dir = os.path.dirname(folder_paths.get_full_path_or_raise("text_encoders", model_path))
        
        mm.cleanup_models_gc()
        mm.unload_all_models()
        # --- A. 模型加载/复用 (时间效率优化) ---
        model_key = (model_dir, dtype)
        

        # --- B. 图像预处理和 OOM 修复 (显存效率优化) ---
       
		# 4. 获取目录中所有图片文件（支持常见格式）
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', 'jfif')
        image_files = [f for f in os.listdir(image_path) 
                      if f.lower().endswith(image_extensions) 
                      and os.path.isfile(os.path.join(image_path, f))]
        
        if not image_files:
            msg = "0 image captioned, 共处理0张图片"
            return {"ui": {"text": (msg,)}, "result": (msg,)}
			
		# --- C. 构造消息和提示词模板 ---
        #if lang == "English":
        #     text_prompt = "Describe this image in detail. Use English"
        #else:
        #     text_prompt = "详细描述这张图片，使用中文"
        prompts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
        if not instruction or not instruction.strip():
            text_prompt = load_prompt_from_file(prompts_file, lang)  # 自动返回多行文本
        else:
            if lang == "中文":
                text_prompt = instruction + "，使用中文"
            elif lang == "English":
                text_prompt = instruction + ". Use English"
            # elif lang == "bbox":
                # text_prompt = instruction + "，返回它们的最小边界框坐标列表。结果必须是一个Python列表的列表，即 [[x1, y1, x2, y2], [x3, y3, x4, y4], ...] 格式。坐标要求：所有坐标值必须是整数。坐标是归一化的，范围是0到1000（表示 0% 到 100% 乘以 10）。每个边界框的顺序为：[左上角X, 左上角Y, 右下角X, 右下角Y]。示例输出：[[250, 150, 450, 500], [600, 700, 800, 950]]请仅输出这个列表结构，不包含任何解释性文字或代码块。"
        print(text_prompt)

        for img_file in image_files:
            try:
                img_path = os.path.join(image_path, img_file)
                # 5.1 读取图片
                pil_raw = Image.open(img_path)
                pil_image = ImageOps.exif_transpose(pil_raw)
                
                # 缓存机制
                result_key = (model_dir, dtype, text_prompt, get_image_hash(pil_image)+str(max_side))
                if result_key in QWEN_RESULT_CACHE:
                    output_text = QWEN_RESULT_CACHE[result_key]
                else:
                    # 确保RGB格式
                    if pil_image.mode == 'RGBA':
                    # 调用 vision_process.py 里的逻辑转为 RGB
                        pil_image = to_rgb(pil_image) 
                    elif pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    # 5.2 图像预处理
                    pil_image_resized = resize_to_limit(pil_image, max_side)
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image_resized},
                                {"type": "text", "text": text_prompt},
                            ]
                        }
                    ]
                    
                    # 按需加载模型
                    if model_key not in QWEN_MODEL_CACHE:
                        #print(f"Qwen3 VL: 首次加载模型 {model_dir}...")
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

                    # 5.4 推理
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        enable_thinking=False,
                    )
                    # image_inputs, video_inputs = process_vision_info(messages)
                    # inputs = self.processor(
                        # text=[text],
                        # images=image_inputs,
                        # videos=video_inputs,
                        # padding=True,
                        # return_tensors="pt",
                    # )
                    inputs = inputs.to(self.model.device)
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=0.7,
                            top_p=0.80,
                            top_k=20,
                            min_p=0.0,
                            #presence_penalty=1.5,
                            repetition_penalty=1.0,
                        )

                    # 5.5 解码结果
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    
                    # Cache the result
                    QWEN_RESULT_CACHE[result_key] = output_text

                    # Limit cache size to prevent memory growth
                    if len(QWEN_RESULT_CACHE) > _batch_cache_size:
                        # Remove oldest entries
                        keys_to_remove = list(QWEN_RESULT_CACHE.keys())[:len(QWEN_RESULT_CACHE) - _batch_cache_size]
                        for key in keys_to_remove:
                            del QWEN_RESULT_CACHE[key]
                
                # 5.6 保存结果为同名txt文件
                txt_filename = os.path.splitext(img_file)[0] + ".txt"
                txt_path = os.path.join(save_path, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(output_text)
                
                count += 1
                print('1', end="")  # 输出后不换行

            except Exception as e:
                continue
        print("")
        
        self.model, self.processor = None, None
        # --- 6. 显存清理 ---
        if not keep_model_loaded:
            QWEN_MODEL_CACHE.clear()
            mm.cleanup_models_gc()
             
        # 强制清理 GPU 缓存 (显存优化)
        gc.collect()
        mm.soft_empty_cache()
        
        # 7. 生成统计结果
        if count > 1:
            stats = f"{count} images captioned, 共处理{count}张图片"
        else:
            stats = f"{count} image captioned, 共处理{count}张图片"

        return {"ui": {"text": (stats,)}, "result": (stats,)}
