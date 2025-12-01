import torch
import numpy as np
import gc
from PIL import Image
from math import ceil
# 必须使用用户指定的类，并确保 BitsAndBytesConfig 导入
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig 
# 假设 vision_process 位于同一目录或可导入
from .vision_process import process_vision_info
import comfy.model_management as mm
import folder_paths
import os
import datetime


# --- 1. Qwen 模型缓存 ---
# 用于存储加载的模型，避免每次节点执行时都重新加载（时间效率优化）
QWEN_MODEL_CACHE = {}


# --- 2. Qwen 模型加载函数 (使用指定的 Qwen3VLForConditionalGeneration) ---
def load_qwen_components(model_dir: str, dtype: str):
    """加载 Qwen 模型、处理器和分词器，支持 4bit 量化。"""

    if dtype == "4bit":
        # 显存优化：使用 4bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
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
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif dtype == "fp8":
        # 显存优化：使用 fp8 量化
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float8_e4m3fn,
            device_map="auto"
        )
    else:
        # 完整精度或 Auto 精度加载
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto"
        )

    # 处理器和分词器
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model.eval()
    return model, processor


# --- 3. 图像预缩放函数 (OOM 关键修复) ---
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

    return image.resize((int(new_width), int(new_height)))


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


# ----------------------------------------------------------------------------------
# --- 4. ComfyUI 节点类 ---
# ----------------------------------------------------------------------------------

class Qwen3Caption:
    def __init__(self):
        # 初始化实例变量，用于存储模型组件
        self.model = None
        self.processor = None
        #self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ), # ComfyUI 的图像输入 Tensor
                "model_path": (folder_paths.get_filename_list("text_encoders"), ),
                "lang": (["中文", "English", "bbox"], {"default": "中文"}),
                "dtype": (["auto", "4bit", "8bit"], {"default": "auto"}), # 强烈建议默认 4bit
                "keep_model_loaded": ("BOOLEAN", {"default": False}), # 默认保持加载
                "max_side": ("INT", {"default": 512, "min": 256, "max": 2240, "step": 32}), # 默认安全尺寸
                #"instruction": ("STRING", {"multiline": True}),
            },
            "optional": {
                "instruction": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "caption"
    CATEGORY = "image/caption"
    OUTPUT_NODE = True


    def caption(self, image: torch.Tensor, model_path: str, lang: str, dtype: str, max_side: int, keep_model_loaded: bool, instruction: str):
        
        if image is None:
            return {"ui": {"text": ("no image, 无图像",)}, "result": ("no image, 无图像",)} 
            
        model_dir = os.path.dirname(folder_paths.get_full_path_or_raise("text_encoders", model_path))
        
        # --- A. 模型加载/复用 (时间效率优化) ---
        cache_key = (model_dir, dtype)
        if cache_key not in QWEN_MODEL_CACHE:
            #print(f"Qwen2.5 VL: 首次加载模型 {model_dir}...")
            try:
                self.model, self.processor = load_qwen_components(model_dir, dtype)
            except Exception as e:
                return {"ui": {"text": ("Failed to load model, 模型加载失败",)}, "result": ("Failed to load model, 模型加载失败",)} 
            QWEN_MODEL_CACHE[cache_key] = (self.model, self.processor)
        else:
            self.model, self.processor = QWEN_MODEL_CACHE[cache_key]

        # --- B. 图像预处理和 OOM 修复 (显存效率优化) ---
        # 1. 处理批次和 Tensor 到 PIL Image 的转换
        image_tensor = image.squeeze(0) # 移除 Batch 维度，形状变为 (H, W, C)
        pil_image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
        # 2. 强制预缩放
        pil_image_resized = resize_to_limit(pil_image, max_side)
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
        
        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": pil_image_resized }, # 使用缩放后的 PIL Image
                    { "type": "text", "text": text_prompt },
                ],
            }
        ]
        # --- D. 预处理、推理和解码 ---
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # 依赖外部 vision_process 导入的函数
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = self.processor(
            # text=[text],
            # images=image_inputs,
            # videos=video_inputs,
            # padding=True,
            # return_tensors="pt",
        # )
        inputs = inputs.to(self.model.device)
        #inputs = inputs.to("cuda")
        with torch.no_grad():
            generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            )
        # 解码并清理
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0] # 取第一个解码结果
        # --- E. 显存清理 ---
        if not keep_model_loaded:
             del self.model, self.processor
             if cache_key in QWEN_MODEL_CACHE:
                 del QWEN_MODEL_CACHE[cache_key]
             self.model, self.processor= None, None
             
        torch.cuda.empty_cache() # 强制清理 GPU 缓存 (显存优化)
        gc.collect()
        mm.soft_empty_cache()
        
        print(output_text)
        return {"ui": {"text": (output_text,)}, "result": (output_text,)} # 必须以元组形式返回
        

class Qwen3CaptionBatch:
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


    def batch_caption(self, model_path: str, lang: str, dtype: str, max_side: int, keep_model_loaded: bool, image_path: str, save_path: str = ""):
        
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
        
        # --- A. 模型加载/复用 (时间效率优化) ---
        cache_key = (model_dir, dtype)
        if cache_key not in QWEN_MODEL_CACHE:
            #print(f"Qwen2.5 VL: 首次加载模型 {model_dir}...")
            try:
                self.model, self.processor = load_qwen_components(model_dir, dtype)
            except Exception as e:
                return {"ui": {"text": ("Failed to load model, 模型加载失败",)}, "result": ("Failed to load model, 模型加载失败",)} 
            QWEN_MODEL_CACHE[cache_key] = (self.model, self.processor)
        else:
             self.model, self.processor = QWEN_MODEL_CACHE[cache_key]

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
                # 5.1 读取图片
                img_path = os.path.join(image_path, img_file)
                pil_image = Image.open(img_path).convert("RGB")  # 确保RGB格式

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

                # 5.4 推理
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
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
                        max_new_tokens=512, 
                    )

                # 5.5 解码结果
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

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
        
        # --- 6. 显存清理 ---
        if not keep_model_loaded:
             del self.model, self.processor
             if cache_key in QWEN_MODEL_CACHE:
                 del QWEN_MODEL_CACHE[cache_key]
             self.model, self.processor= None, None
             
        torch.cuda.empty_cache() # 强制清理 GPU 缓存 (显存优化)
        gc.collect()
        mm.soft_empty_cache()
        
        # 7. 生成统计结果
        if count > 1:
            stats = f"{count} images captioned, 共处理{count}张图片"
        else:
            stats = f"{count} image captioned, 共处理{count}张图片"

        return {"ui": {"text": (stats,)}, "result": (stats,)}
