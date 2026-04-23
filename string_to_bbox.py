import json
import re
import math
from typing import List, Tuple

class StringToBbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {
                    "multiline": True,
                    #"default": '[\n    {"bbox_2d": [284, 252, 488, 815], "label": "凌波丽"}\n]',
                    #"placeholder": "输入JSON格式的bbox字符串"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    #"description": "图片的实际宽度（像素）"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    #"description": "图片的实际高度（像素）"
                }),
                "index": ("STRING", {
                    "default": "",
                    #"placeholder": "指定要提取的bbox索引，多个用逗号分隔（如0,1,2）"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "BBOX")
    RETURN_NAMES = ("center_coordinates", "bboxes")
    FUNCTION = "convert_to_bbox"
    CATEGORY = "image/caption"
    #DESCRIPTION = "将JSON格式的千分比bbox字符串转换为实际像素坐标的BBOX"
    
    def _clean_and_extract_coords(self, input_string: str) -> List[List[float]]:
        all_bboxes = []
        
        # 1. 使用更稳健的正则：直接从原始字符串中提取所有非嵌套的 [ ... ] 内容。
        # 正则 r'\[([^\[\]]+)\]' 匹配 [ 后面跟着任何非 [ 或 ] 的字符，直到下一个 ]。
        # 这样可以忽略包含 JSON 键值对的外层结构，专门提取内层的数字列表。
        # 这样可以同时处理：
        #   - 纯数组输入：[[250, 150, 450, 500], [600, 700, 800, 950]]
        #   - JSON 对象输入：{"bbox_2d": [284, 252, 488, 815], ...}
        bracket_groups = re.findall(r'\[([^\[\]]+)\]', input_string, re.DOTALL) #
        
        # 2. 解析每个数组，筛选有效bbox（长度为4的数值数组）
        for group in bracket_groups:
            # 针对捕获的内容进行局部清理，只保留数字、逗号、小数点、负号
            # 这样可以在不破坏整体结构的情况下，清理掉例如 "label" 或其他文本
            cleaned_group = re.sub(r'[^-0-9,.]', '', group)
            
            # 分割数值字符串，过滤空字符串
            num_strings = [s.strip() for s in cleaned_group.split(',') if s.strip()]
            nums = []
            for num_str in num_strings:
                # 验证是否为有效数字（整数/小数）
                if re.match(r'^-?\d+(\.\d+)?$', num_str):
                    nums.append(float(num_str))
            
            # 只保留长度为4的数组（符合bbox坐标格式）
            if len(nums) == 4:
                # 去重
                if nums not in all_bboxes:
                    all_bboxes.append(nums)
        
        # 调试输出仅用于检查最终结果
        # print(f"清理后的字符串：{input_string.strip()}") # 保持原始字符串
        
        return all_bboxes

    def convert_to_bbox(self, string: str, width: int, height: int, index: str) -> Tuple[str, List[List[int]]]:
        """
        核心转换函数：清理字符串→提取坐标→按规则转换为像素坐标
        """
        # 初始化返回值
        center_points = []
        bboxes = []
        
        try:
            # 1. 清理字符串并提取所有有效的bbox坐标数组
            all_bboxes = self._clean_and_extract_coords(string)
            if not all_bboxes:
                raise ValueError("未从输入字符串中提取到有效的bbox坐标（需包含长度为4的数值数组）")
            
            # 2. 处理索引参数
            if index.strip():
                # 解析用户指定的索引（支持多个，用逗号分隔）
                indexes = []
                for idx_str in index.split(","):
                    idx_str = idx_str.strip()
                    if idx_str.isdigit():
                        indexes.append(int(idx_str))
                    else:
                        raise ValueError(f"无效的索引值：{idx_str}（必须是整数）")
            else:
                # 如果索引为空，使用所有可用的bbox索引
                indexes = list(range(len(all_bboxes)))
            
            # 3. 遍历指定的索引，转换坐标
            for idx in indexes:
                if 0 <= idx < len(all_bboxes):
                    # 获取千分比坐标（左上x, 左上y, 右下x, 右下y）
                    x1_percent, y1_percent, x2_percent, y2_percent = all_bboxes[idx]
                    
                    # 转换为实际像素坐标（四舍五入为整数）
                    x1 = int(math.floor(x1_percent * width / 1000))
                    y1 = int(math.floor(y1_percent * height / 1000))
                    x2 = int(math.ceil(x2_percent * width / 1000))
                    y2 = int(math.ceil(y2_percent * height / 1000))
                    
                    # 确保坐标在合理范围内（不小于0，不大于图片尺寸）
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(x1, min(x2, width))  # 确保x2 >= x1
                    y2 = max(y1, min(y2, height))  # 确保y2 >= y1
                    
                    # 计算中心点坐标
                    center_x = int(round((x1 + x2) / 2))
                    center_y = int(round((y1 + y2) / 2))
                    
                    # 添加到结果列表
                    center_points.append({"x": center_x, "y": center_y})
                    bboxes.append([x1, y1, x2, y2])
                else:
                    raise ValueError(f"索引{idx}超出范围（有效范围：0-{len(all_bboxes)-1}）")
        
        except json.JSONDecodeError as e:
            # JSON解析错误
            print(f"JSON解析错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        except ValueError as e:
            # 自定义错误
            print(f"参数错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        except Exception as e:
            # 其他未知错误
            print(f"转换错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        
        # 输出调试信息
        #print(f"提取到的原始坐标（千分比）：{all_bboxes}")
        #print(f"转换后的BBOXes（像素）：{bboxes}")
        #print(f"中心点坐标：{center_points}")
        
        # 返回结果（中心点JSON字符串 + BBOX列表）
        return (json.dumps(center_points), bboxes)


class StringToComfyBbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {
                    "multiline": True,
                    #"default": '[\n    {"bbox_2d": [284, 252, 488, 815], "label": "凌波丽"}\n]',
                    #"placeholder": "输入JSON格式的bbox字符串"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    #"description": "图片的实际宽度（像素）"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    #"description": "图片的实际高度（像素）"
                }),
                "index": ("STRING", {
                    "default": "",
                    #"placeholder": "指定要提取的bbox索引，多个用逗号分隔（如0,1,2）"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "BOUNDING_BOX")
    RETURN_NAMES = ("center_coordinates", "bboxes")
    FUNCTION = "convert_to_comfy_bbox"
    CATEGORY = "image/caption"
    #DESCRIPTION = "将JSON格式的千分比bbox字符串转换为实际像素坐标的BBOX"
    
    def _clean_and_extract_coords(self, input_string: str) -> List[List[float]]:
        all_bboxes = []
        
        # 1. 使用更稳健的正则：直接从原始字符串中提取所有非嵌套的 [ ... ] 内容。
        # 正则 r'\[([^\[\]]+)\]' 匹配 [ 后面跟着任何非 [ 或 ] 的字符，直到下一个 ]。
        # 这样可以忽略包含 JSON 键值对的外层结构，专门提取内层的数字列表。
        # 这样可以同时处理：
        #   - 纯数组输入：[[250, 150, 450, 500], [600, 700, 800, 950]]
        #   - JSON 对象输入：{"bbox_2d": [284, 252, 488, 815], ...}
        bracket_groups = re.findall(r'\[([^\[\]]+)\]', input_string, re.DOTALL) #
        
        # 2. 解析每个数组，筛选有效bbox（长度为4的数值数组）
        for group in bracket_groups:
            # 针对捕获的内容进行局部清理，只保留数字、逗号、小数点、负号
            # 这样可以在不破坏整体结构的情况下，清理掉例如 "label" 或其他文本
            cleaned_group = re.sub(r'[^-0-9,.]', '', group)
            
            # 分割数值字符串，过滤空字符串
            num_strings = [s.strip() for s in cleaned_group.split(',') if s.strip()]
            nums = []
            for num_str in num_strings:
                # 验证是否为有效数字（整数/小数）
                if re.match(r'^-?\d+(\.\d+)?$', num_str):
                    nums.append(float(num_str))
            
            # 只保留长度为4的数组（符合bbox坐标格式）
            if len(nums) == 4:
                # 去重
                if nums not in all_bboxes:
                    all_bboxes.append(nums)
        
        # 调试输出仅用于检查最终结果
        # print(f"清理后的字符串：{input_string.strip()}") # 保持原始字符串
        
        return all_bboxes

    def convert_to_comfy_bbox(self, string: str, width: int, height: int, index: str) -> Tuple[str, List[List[int]]]:
        """
        核心转换函数：清理字符串→提取坐标→按规则转换为像素坐标
        """
        # 初始化返回值
        center_points = []
        bboxes = []
        
        try:
            # 1. 清理字符串并提取所有有效的bbox坐标数组
            all_bboxes = self._clean_and_extract_coords(string)
            if not all_bboxes:
                raise ValueError("未从输入字符串中提取到有效的bbox坐标（需包含长度为4的数值数组）")
            
            # 2. 处理索引参数
            if index.strip():
                # 解析用户指定的索引（支持多个，用逗号分隔）
                indexes = []
                for idx_str in index.split(","):
                    idx_str = idx_str.strip()
                    if idx_str.isdigit():
                        indexes.append(int(idx_str))
                    else:
                        raise ValueError(f"无效的索引值：{idx_str}（必须是整数）")
            else:
                # 如果索引为空，使用所有可用的bbox索引
                indexes = list(range(len(all_bboxes)))
            
            # 3. 遍历指定的索引，转换坐标
            for idx in indexes:
                if 0 <= idx < len(all_bboxes):
                    # 获取千分比坐标（左上x, 左上y, 右下x, 右下y）
                    x1_percent, y1_percent, x2_percent, y2_percent = all_bboxes[idx]
                    
                    # 转换为实际像素坐标（四舍五入为整数）
                    x1 = int(math.floor(x1_percent * width / 1000))
                    y1 = int(math.floor(y1_percent * height / 1000))
                    x2 = int(math.ceil(x2_percent * width / 1000))
                    y2 = int(math.ceil(y2_percent * height / 1000))
                    
                    # 确保坐标在合理范围内（不小于0，不大于图片尺寸）
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(x1, min(x2, width))  # 确保x2 >= x1
                    y2 = max(y1, min(y2, height))  # 确保y2 >= y1
                    
                    # 计算中心点坐标
                    center_x = int(round((x1 + x2) / 2))
                    center_y = int(round((y1 + y2) / 2))
                    
                    # 添加到结果列表
                    center_points.append({"x": center_x, "y": center_y})
                    bboxes.append({"x": x1, "y": y1, "width": max(x2-x1,1), "height": max(y2-y1,1)})
                else:
                    raise ValueError(f"索引{idx}超出范围（有效范围：0-{len(all_bboxes)-1}）")
        
        except json.JSONDecodeError as e:
            # JSON解析错误
            print(f"JSON解析错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [{"x": 0, "y": 0, "width": 0, "height": 0}]
        except ValueError as e:
            # 自定义错误
            print(f"参数错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [{"x": 0, "y": 0, "width": 0, "height": 0}]
        except Exception as e:
            # 其他未知错误
            print(f"转换错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [{"x": 0, "y": 0, "width": 0, "height": 0}]
        
        # 输出调试信息
        #print(f"提取到的原始坐标（千分比）：{all_bboxes}")
        #print(f"转换后的BBOXes（像素）：{bboxes}")
        #print(f"中心点坐标：{center_points}")
        
        # 返回结果（中心点JSON字符串 + BBOX列表）
        return (json.dumps(center_points), bboxes)


class StringToSam3Box:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {
                    "multiline": True,
                    }),
                #"bboxes": ("BBOX",),  # 接收 StringToBbox 的输出
                # "width": ("INT", {
                    # "default": 1024,
                    # "min": 1,
                    # "max": 10000,
                    # "step": 1,
                    # "description": "原图宽度，用于归一化"
                # }),
                # "height": ("INT", {
                    # "default": 1024,
                    # "min": 1,
                    # "max": 10000,
                    # "step": 1,
                    # "description": "原图高度，用于归一化"
                # }),
                "label_type": (["positive", "negative"], {
                    "default": "positive"
                }),
                "index": ("STRING", {
                    "default": "",
                })
            },
        }
    
    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_BOXES_PROMPT",)
    RETURN_NAMES = ("point_prompt", "box_prompt",)
    FUNCTION = "convert_to_sam3box"
    CATEGORY = "image/caption"
    
    def _clean_and_extract_coords(self, input_string: str) -> List[List[float]]:
        all_bboxes = []
        
        # 1. 使用更稳健的正则：直接从原始字符串中提取所有非嵌套的 [ ... ] 内容。
        # 正则 r'\[([^\[\]]+)\]' 匹配 [ 后面跟着任何非 [ 或 ] 的字符，直到下一个 ]。
        # 这样可以忽略包含 JSON 键值对的外层结构，专门提取内层的数字列表。
        # 这样可以同时处理：
        #   - 纯数组输入：[[250, 150, 450, 500], [600, 700, 800, 950]]
        #   - JSON 对象输入：{"bbox_2d": [284, 252, 488, 815], ...}
        bracket_groups = re.findall(r'\[([^\[\]]+)\]', input_string, re.DOTALL) #
        
        # 2. 解析每个数组，筛选有效bbox（长度为4的数值数组）
        for group in bracket_groups:
            # 针对捕获的内容进行局部清理，只保留数字、逗号、小数点、负号
            # 这样可以在不破坏整体结构的情况下，清理掉例如 "label" 或其他文本
            cleaned_group = re.sub(r'[^-0-9,.]', '', group)
            
            # 分割数值字符串，过滤空字符串
            num_strings = [s.strip() for s in cleaned_group.split(',') if s.strip()]
            nums = []
            for num_str in num_strings:
                # 验证是否为有效数字（整数/小数）
                if re.match(r'^-?\d+(\.\d+)?$', num_str):
                    nums.append(float(num_str))
            
            # 只保留长度为4的数组（符合bbox坐标格式）
            if len(nums) == 4:
                # 去重
                if nums not in all_bboxes:
                    all_bboxes.append(nums)
        
        # 调试输出仅用于检查最终结果
        # print(f"清理后的字符串：{input_string.strip()}") # 保持原始字符串
        
        return all_bboxes
    
    def convert_to_sam3box(self, string, label_type, index):
        sam_labels = []
        # 确定标签的布尔值
        is_positive = False if label_type == "negative" else True
        
        sam_boxes = []
        center_points = []
        
        try:
            # 1. 清理字符串并提取所有有效的bbox坐标数组
            all_bboxes = self._clean_and_extract_coords(string)
            if not all_bboxes:
                raise ValueError("未从输入字符串中提取到有效的bbox坐标（需包含长度为4的数值数组）")
            
            # 2. 处理索引参数
            if index.strip():
                # 解析用户指定的索引（支持多个，用逗号分隔）
                indexes = []
                for idx_str in index.split(","):
                    idx_str = idx_str.strip()
                    if idx_str.isdigit():
                        indexes.append(int(idx_str))
                    else:
                        raise ValueError(f"无效的索引值：{idx_str}（必须是整数）")
            else:
                # 如果索引为空，使用所有可用的bbox索引
                indexes = list(range(len(all_bboxes)))
            
            # 3. 遍历指定的索引，转换坐标
            for idx in indexes:
                if 0 <= idx < len(all_bboxes):
                    # 获取千分比坐标（左上x, 左上y, 右下x, 右下y）
                    x1_percent, y1_percent, x2_percent, y2_percent = all_bboxes[idx]
                    
                    # 转换为百分比像素坐标（小数）
                    if x2_percent > 1 or y2_percent > 1:
                        x1 = x1_percent / 1000
                        y1 = y1_percent / 1000
                        x2 = x2_percent / 1000
                        y2 = y2_percent / 1000
                    
                    # 确保坐标在合理范围内（不小于0，不大于1）
                    x1 = max(0, min(x1, 1))
                    y1 = max(0, min(y1, 1))
                    x2 = max(0, min(x2, 1))
                    y2 = max(0, min(y2, 1))
                    
                    # 计算中心点坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    box_w = abs(x2 - x1)
                    box_h = abs(y2 - y1)
                    
                    # 添加到结果列表
                    #center_points.append({"x": center_x, "y": center_y})
                    center_points.append([center_x, center_y])
                    sam_boxes.append([center_x, center_y, box_w, box_h])
                    sam_labels.append(is_positive)
                else:
                    raise ValueError(f"索引{idx}超出范围（有效范围：0-{len(all_bboxes)-1}）")
        
        except json.JSONDecodeError as e:
            # JSON解析错误
            print(f"JSON解析错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        except ValueError as e:
            # 自定义错误
            print(f"参数错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        except Exception as e:
            # 其他未知错误
            print(f"转换错误：{str(e)}")
            center_points = [{"x": 0, "y": 0}]
            bboxes = [[0, 0, 0, 0]]
        
        # for bbox in bboxes:
            # # bbox 格式通常为 [x1, y1, x2, y2] (像素值)
            # x1, y1, x2, y2 = bbox
            
            # # 计算宽度和高度 (像素)
            # box_w = abs(x2 - x1)
            # box_h = abs(y2 - y1)
            
            # # 计算中心点 (像素)
            # center_x = (x1 + x2) / 2
            # center_y = (y1 + y2) / 2
            
            # # 归一化 (转换为 0.0 - 1.0)
            # # 格式: [center_x, center_y, width, height]
            # norm_cx = center_x / width
            # norm_cy = center_y / height
            # norm_w = box_w / width
            # norm_h = box_h / height
            
            # # 确保数值在 0-1 之间 (防止因为浮点数计算超出极小范围)
            # norm_cx = max(0.0, min(1.0, norm_cx))
            # norm_cy = max(0.0, min(1.0, norm_cy))
            # norm_w = max(0.0, min(1.0, norm_w))
            # norm_h = max(0.0, min(1.0, norm_h))
            
            #sam_boxes.append([norm_cx, norm_cy, norm_w, norm_h])
        
            
        # 构建最终字典
        result_boxes = {
            "boxes": sam_boxes,
            "labels": sam_labels
        }
        result_points = {
            "points": center_points,
            "labels": sam_labels
        }
        
        #return (json.dumps(result, indent=4),)
        return (result_points, result_boxes)