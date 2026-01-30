"""
在训练脚本中使用自定义Qwen2 forward的示例
"""

# 在导入模型之前先应用patch
from custom_qwen2_forward import apply_custom_qwen2_forward

# 应用自定义forward函数
apply_custom_qwen2_forward()

# 然后正常导入和使用模型
from transformers import AutoModel, AutoTokenizer

# 现在所有的Qwen2模型都会使用您的自定义forward函数
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")

# 测试
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)  # 这会调用您的自定义forward函数

print("模型运行完成!")