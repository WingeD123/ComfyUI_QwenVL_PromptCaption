import torch
import numpy as np
import gc
from PIL import Image
from math import ceil
# 必须使用用户指定的类，并确保 BitsAndBytesConfig 导入
from transformers import AutoModelForCausalLM, BitsAndBytesConfig 

import comfy.model_management as mm
import folder_paths
import os
import datetime
import re


# --- 1. Qwen 模型缓存 ---
# 用于存储加载的模型，避免每次节点执行时都重新加载（时间效率优化）
QWEN_MODEL_CACHE = {}


# --- 2. Qwen 模型加载函数 (使用指定的 Qwen2_5_VLForConditionalGeneration) ---
def load_ovis_components(model_dir: str, dtype: str):
    """加载 Qwen 模型、处理器和分词器，支持 4bit 量化。"""
    
    if dtype == "4bit":
        # 显存优化：使用 4bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif dtype == "8bit":
        # 显存优化：使用 8bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            #bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif dtype == "fp8":
        # 显存优化：使用 fp8 量化
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float8_e4m3fn,
            device_map="auto",
            trust_remote_code=True,
        )
    elif dtype == "bf16":
        # 显存优化：使用 bf16 量化
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # 完整精度或 Auto 精度加载
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    # 处理器和分词器
    #processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model.eval()
    return model#, processor


# --- 3. 图像预缩放函数 (OOM 关键修复) ---
def resize_to_limit(image: Image.Image, max_side: int):
    """强制将图像最大边长缩放到指定限制，并确保是 Qwen 所需的 28 的倍数。"""
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

class Ovis25Run:
    def __init__(self):
        # 初始化实例变量，用于存储模型组件
        self.model = None
        #self.processor = None
        #self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ), # ComfyUI 的图像输入 Tensor
                "model_path": (folder_paths.get_filename_list("text_encoders"), ),
                "lang": (["中文", "English", "bbox"], {"default": "中文"}),
                "dtype": (["auto", "4bit", "8bit", "bf16"], {"default": "bf16"}), # 强烈建议默认 4bit
                "keep_model_loaded": ("BOOLEAN", {"default": False}), # 默认保持加载
                "thinking": ("BOOLEAN", {"default": True}), # enable_thinking
                #"max_side": ("INT", {"default": 532, "min": 252, "max": 2240, "step": 28}), # 默认安全尺寸
                "instruction": ("STRING", {"multiline": True}),
            },
            # "optional": {
                # "instruction": ("STRING", {"multiline": True}),
            # }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "full_output")
    FUNCTION = "run"
    CATEGORY = "image/caption"
    OUTPUT_NODE = True


    def run(self, image: torch.Tensor, model_path: str, lang: str, dtype: str, keep_model_loaded: bool, thinking: bool, instruction: str):
        
        if image is None:
            return {"ui": {"text": ("no image, 无图像",)}, "result": ("no image, 无图像",)} 
            
        model_dir = os.path.dirname(folder_paths.get_full_path_or_raise("text_encoders", model_path))
        
        # --- A. 模型加载/复用 (时间效率优化) ---
        cache_key = (model_dir, dtype)
        if cache_key not in QWEN_MODEL_CACHE:
            #print(f"Qwen2.5 VL: 首次加载模型 {model_dir}...")
            try:
                self.model = load_ovis_components(model_dir, dtype)
            except Exception as e:
                return {"ui": {"text": ("Failed to load model, 模型加载失败",)}, "result": ("Failed to load model, 模型加载失败",)} 
            QWEN_MODEL_CACHE[cache_key] = (self.model, None)
        else:
            self.model, _ = QWEN_MODEL_CACHE[cache_key]
        
        # --- B. 图像预处理和 OOM 修复 (显存效率优化) ---
        # 1. 处理批次和 Tensor 到 PIL Image 的转换
        image_tensor = image.squeeze(0) # 移除 Batch 维度，形状变为 (H, W, C)
        pil_image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))

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
        #text_prompt = text_prompt + ". End your response with 'Final answer: '."
        print(text_prompt)
        
        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": pil_image },
                    { "type": "text", "text": text_prompt },
                ],
            }
        ]
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        input_ids = input_ids.cuda()
        model_dtype = next(self.model.parameters()).dtype
        #print(model_dtype)
        # 将 pixel_values 转换为和模型一致的 dtype
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=thinking,
                enable_thinking_budget=True,# Only effective if enable_thinking is True.
                max_new_tokens=3072,# Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
                thinking_budget=2048,
            )

        output_text = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        ### 1. 提取 <think> 标签内的内容
        # 使用非贪婪匹配 `.*?` 来匹配 <think> 和 </think> 之间的所有内容
        # `re.DOTALL` (或 `re.S`) 标志让 `.` 也能匹配换行符
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, output_text, re.DOTALL)

        if think_match:
            # group(1) 提取的是第一个捕获组 (即括号 `(...)` 里的内容)
            think_text = think_match.group(1).strip() # .strip() 用于去除首尾的空白字符和换行符
        else:
            think_text = ""

        ### 2. 提取 <think> 标签外的其他内容
        # 使用 `re.sub()` 将 <think>... </think> 整个匹配项替换为空字符串
        # 注意，这里使用相同的模式，并确保匹配换行符
        other_text = re.sub(think_pattern, "", output_text, flags=re.DOTALL).strip()
        
        # --- E. 显存清理 ---
        if not keep_model_loaded:
             del self.model
             if cache_key in QWEN_MODEL_CACHE:
                 del QWEN_MODEL_CACHE[cache_key]
             self.model = None
             
        torch.cuda.empty_cache() # 强制清理 GPU 缓存 (显存优化)
        gc.collect()
        mm.soft_empty_cache()
        
        print(output_text)
        return {"ui": {"text": (output_text,)}, "result": (other_text, output_text)} # 必须以元组形式返回
