from .qwen_25 import Qwen25Caption, Qwen25CaptionBatch
from .qwen_3 import Qwen3Caption, Qwen3CaptionBatch
from .string_to_bbox import StringToBbox

# ----------------------------------------------------------------------------------
# --- ComfyUI 映射 ---
# ----------------------------------------------------------------------------------

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "Qwen25Caption": Qwen25Caption,
    "Qwen25CaptionBatch": Qwen25CaptionBatch,
    "Qwen3Caption": Qwen3Caption,
    "Qwen3CaptionBatch": Qwen3CaptionBatch,
    "StringToBbox": StringToBbox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen25Caption": "Qwen2.5 VL Caption (Inverse Prompt)",
    "Qwen25CaptionBatch": "Qwen2.5 VL Batch Caption",
    "Qwen3Caption": "Qwen3 VL Caption (Inverse Prompt)",
    "Qwen3CaptionBatch": "Qwen3 VL Batch Caption",
    "StringToBbox": "String to BBOX"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]