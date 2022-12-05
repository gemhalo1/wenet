import logging
import traceback
import onnx
import onnxruntime
import numpy as np

from onnxruntime.quantization.quantize import quantize_dynamic
from onnxruntime.quantization.quant_utils import QuantFormat

logger = logging.getLogger(__name__)

input_model_path = '/home/gem/development/learn/wenet_stream_model_0830/model_16_4_dynamic/encoder.onnx'
quant_model_path = '/home/gem/development/learn/wenet_stream_model_0830/model_16_4_dynamic_quant/encoder.onnx'

from onnxruntime.quantization.quant_utils import QuantType
quantize_dynamic(input_model_path,
                 quant_model_path,

                 # ConvInteger only supports UInt8
                 weight_type=QuantType.QUInt8,

                 extra_options={
                     'ActivationSymmetric': True,
                     'WeightSymmetric': True,
                 }
                 )

from onnx import shape_inference
xxx = onnx.load(quant_model_path)
ooo = shape_inference.infer_shapes(xxx, check_type=True, strict_mode=True, data_prop=True)
onnx.save(ooo, quant_model_path)
