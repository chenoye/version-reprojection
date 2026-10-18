import onnx
from onnx import version_converter

print("正在加载模型...")
model = onnx.load("armor_det.onnx")
print(f"原始 Opset 版本: {model.opset_import[0].version}")

print("正在转换版本...")
converted_model = version_converter.convert_version(model, 21)
print(f"转换后 Opset 版本: {converted_model.opset_import[0].version}")

onnx.save(converted_model, "armor_det_opset21.onnx")
print("模型转换完成！已生成 armor_det_opset21.onnx")