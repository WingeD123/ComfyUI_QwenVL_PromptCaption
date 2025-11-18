import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";


app.registerExtension({
	// 注册扩展的唯一名称
	name: "caption.qwen.result.display", 
	
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// 1. 识别您的目标节点名称
		const TARGET_NODES = ["Qwen25Caption", "Qwen25CaptionBatch", "Qwen3Caption", "Qwen3CaptionBatch"];
		if (TARGET_NODES.includes(nodeData.name)) {
			
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
                
			// 根据节点类型设置不同的 widget 名称和高度（便于区分）
			const nameMap = {
			    "Qwen25Caption": "caption_result_2.5",
			    "Qwen25CaptionBatch": "batch_summary_2.5",
			    "Qwen3Caption": "caption_result_3",
			    "Qwen3CaptionBatch": "batch_summary_3"
			};
			// 使用查找表，如果找不到匹配项，则默认值为 ""
			const widgetName = nameMap[nodeData.name] || "";
			
			//const height = nodeData.name === "Qwen25Caption" ? "120px" : "80px";

				// 2. 清理旧的 Widget
				if (this.widgets) {
					// 查找并移除我们之前创建的 widget
					const pos = this.widgets.findIndex((w) => w.name === widgetName);
					if (pos !== -1) {
						for (let i = pos; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = pos;
					}
				}
				
				// 3. 动态创建新的文本框 Widget
				if (message.text && message.text.length > 0) {
                    
                    // ComfyWidgets["STRING"] 用于创建多行文本输入框
					const w = ComfyWidgets["STRING"](this, widgetName, ["STRING", { multiline: true }], app).widget;
					
                    // 设置样式和属性
					w.inputEl.readOnly = true; 
					w.inputEl.style.opacity = 0.7; 
                    //w.inputEl.style.height = height; // 应用不同高度
					
                    // 4. 将后端返回的结果填充到文本框中
					w.value = message.text[0]; // 始终取 message.text 数组的第一个元素

					// 5. 调整节点大小
					//this.onResize?.(this.size);
				}
			};
		}
	},
});

