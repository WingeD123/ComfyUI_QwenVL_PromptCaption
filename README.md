# ComfyUI_QwenVL_PromptCaption
Leverages Qwen 2.5/3 VL for prompt inversion &amp; caption generation in ComfyUI

---

## 重要说明 | Important Note
❌ 插件不自动下载模型，可复用 ComfyOrg 提供的 `qwen_2.5_vl_7b.safetensors`，也可手动下载其它Qwen VL模型。  
❌ This plugin does not auto-download models. It can reuse `qwen_2.5_vl_7b.safetensors` provided by ComfyOrg, or manually download other Qwen VL models.

---

## 节点 | Nodes
1. Qwen 2.5 VL Caption: Single image prompt inversion  
   Qwen 2.5 VL Caption：单图提示词反推
2. Qwen 2.5 VL Batch Caption: Batch image prompt inversion (folder input)  
   Qwen 2.5 VL Batch Caption：目录批量图片提示词反推
3. Qwen 3 VL Caption: Single image prompt inversion  
   Qwen 3 VL Caption：单图提示词反推
4. Qwen 3 VL Batch Caption: Batch image prompt inversion (folder input)  
   Qwen 3 VL Batch Caption：目录批量图片提示词反推
5. Ovis 2.5 Run: Run Ovis 2.5 model  
   Ovis 2.5 Run：运行 Ovis 2.5 模型
<img width="1294" height="875" alt="nodes1" src="https://github.com/user-attachments/assets/be0e7a0d-906e-4630-b920-72fc7dfe598f" />

---

## 安装方法 | Installation
a. Via ComfyUI Manager  
   通过 ComfyUI Manager 安装  
b. Manual install:  
   手动安装：
   1. Copy the plugin folder to `ComfyUI/custom_nodes/`  
      复制插件目录至 `ComfyUI/custom_nodes/`  
   2. Update dependency: `transformers>=4.57.0`  
      更新依赖：`transformers>=4.57.0`

---

## 使用方法 | Usage
1. Download the model  
   下载模型  
2. Edit prompt templates (optional)  
   编辑指令提示词（可选）  
3. Adjust node inputs  
   调整节点输入参数  
4. Click "Run"  
   点击运行

---

## 模型说明 | Model Notes
- 模型读取路径：ComfyUI 的 `text_encoders` 目录（需手动放置已下载模型）。  
  Model path: ComfyUI's `text_encoders` folder (place downloaded models manually).  

### 复用 ComfyOrg 模型 | Reuse ComfyOrg Model
To reuse `qwen_2.5_vl_7b.safetensors`:  
复用 `qwen_2.5_vl_7b.safetensors` 步骤：
1. Create a FOLDER in ComfyUI/models/text_encoders  
   在ComfyUI/models/text_encoders中创建一个文件夹
2. Rename the model file to `model.safetensors` and move it into the FOLDER  
   将模型文件重命名为 `model.safetensors`并移入创建的文件夹  
3. Add required config files (from Qwen 2.5 VL's official Hugging Face repo)  
   添加必要配置文件（取自 Qwen 2.5 VL 官方 Hugging Face 仓库）
   https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
<img width="834" height="345" alt="nodes2" src="https://github.com/user-attachments/assets/80f9f42c-a71e-45ca-9b88-9c9c5567508c" />

✅ No extra disk usage – model remains usable for ComfyUI's Qwen Image/Edit model.  
✅ 无额外硬盘消耗，不影响原模型用于 ComfyUI 的 Qwen Image/Edit模型。

### 直接下载官方模型 | Direct Download
Download Qwen 2.5/3 VL official repo from Hugging Face, then place it in `text_encoders`.  
从 Hugging Face 下载 Qwen 2.5/3 VL 官方仓库，直接放入 `text_encoders` 目录即可。

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

国内也可从网盘下载：https://pan.quark.cn/s/b3975e789c3c

Support Ovis 2.5 models
新支持 Ovis 2.5 模型
https://huggingface.co/AIDC-AI/Ovis2.5-2B
https://huggingface.co/AIDC-AI/Ovis2.5-9B

---

## 自定义提示词 | Custom Prompts
Now you can input instruction directly, or  
现在可以直接输入指令，或者  
Edit `prompts.txt` in the custom_nodes folder (follow the existing format):  
修改插件目录下的 `prompts.txt` 文件（参考原有格式）：
- Support multiple prompts  
  支持多条提示词  
- The nodes will use the last prompt matching the language  
  自动读取对应语言的最后一条提示词

---

## 模型精度建议 | VRAM & Precision Recommendations
| 显存 (VRAM) | 推荐精度 (Recommended Precision) |
|-------------|----------------------------------|
| 6-8GB       | Qwen 2.5 VL 7B (4bit) / Qwen 3 VL 8B (4bit) / Qwen 3 VL 4B (8bit) |
| 10-16GB     | Qwen 2.5 VL 7B (8bit) / Qwen 3 VL 8B (8bit) / Qwen 3 VL 4B (bf16) |
| 16GB+       | bf16 (full precision)            |

---

## 参数说明 | Parameter Notes
### `keep_model_loaded`
- Use True to Keep model in VRAM for consecutive prompt inversion tasks  
  连续进行提示词反推时选 True  
- False won't impact performance during batch node run  
  批量节点选 False仅在全部图片处理完成后清理模型，不影响过程性能

### `max_side`
- Pre-scales the image's longer side to this size  
  预缩放图片长边尺寸  
- Larger values may reduce processing speed  
  设置过大会导致速度下降  

### `save_path`
- will use image_path to save output if save_path not set  
  save_path为空时会使用image_path保存输出  
