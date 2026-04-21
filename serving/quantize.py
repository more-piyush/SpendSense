from onnxruntime.quantization import quantize_dynamic, QuantType
import os
input_path = 'onnx_model/model.onnx'
output_path = 'onnx_model_quantized/model.onnx'
os.makedirs('onnx_model_quantized', exist_ok=True)
print(f'Quantizing {input_path} -> {output_path}')
quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)
print('Quantization complete.')
