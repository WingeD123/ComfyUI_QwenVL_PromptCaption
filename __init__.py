from .qwen_25 import Qwen25Caption, Qwen25CaptionBatch
from .qwen_3 import Qwen3Caption, Qwen3CaptionBatch
from .qwen_35 import Qwen35Caption, Qwen35CaptionBatch
from .string_to_bbox import StringToBbox, StringToSam3Box
from .ovis_25 import Ovis25Run
from .asid_captioner import ASID_Caption

# ----------------------------------------------------------------------------------
# --- ComfyUI 映射 ---
# ----------------------------------------------------------------------------------

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "Qwen25Caption": Qwen25Caption,
    "Qwen25CaptionBatch": Qwen25CaptionBatch,
    "Qwen3Caption": Qwen3Caption,
    "Qwen3CaptionBatch": Qwen3CaptionBatch,
    "StringToBbox": StringToBbox,
    "StringToSam3Box": StringToSam3Box,
    "Ovis25Run": Ovis25Run,
    "ASID_Caption": ASID_Caption,
    "Qwen35Caption": Qwen35Caption,
    "Qwen35CaptionBatch": Qwen35CaptionBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen25Caption": "Qwen2.5 VL Caption (Inverse Prompt)",
    "Qwen25CaptionBatch": "Qwen2.5 VL Batch Caption",
    "Qwen3Caption": "Qwen3 VL Caption (Inverse Prompt)",
    "Qwen3CaptionBatch": "Qwen3 VL Batch Caption",
    "StringToBbox": "String to BBOX",
    "StringToSam3Box": "String to SAM3 Box",
    "Ovis25Run": "Ovis2.5 Run",
    "ASID_Caption": "ASID Captioner (Inverse Prompt)",
    "Qwen35Caption": "Qwen3.5 VL Caption (Inverse Prompt)",
    "Qwen35CaptionBatch": "Qwen3.5 VL Batch Caption",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]