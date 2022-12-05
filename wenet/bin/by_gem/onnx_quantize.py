from onnxruntime.quantization.quantize import quantize_dynamic
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

for model in ['encoder.onnx', 'decoder.onnx', 'ctc.onnx']:
    print(f'exporting quantized model: {model}')
    quantize_dynamic(f"/home/gem/development/learn/wenet_stream_model_1118/model_8_all/{model}",
                     f"/home/gem/development/learn/wenet_stream_model_1118/model_8_all_q/{model}",
                     weight_type=QuantType.QUInt8
                     )
