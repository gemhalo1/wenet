import onnx
from onnx import shape_inference
def convert_dynamic(onnx_path, output_path, override_dynamic_inputs):
    onnx_model = shape_inference.infer_shapes(onnx.load(onnx_path), check_type=True, strict_mode=True, data_prop=True)
    onnx_graph = onnx_model.graph

    if override_dynamic_inputs is not None:
        for name in override_dynamic_inputs.keys():
            override_info = override_dynamic_inputs.get(name)

            name_found = False
            for t in onnx_graph.input:
                if t.name == name:
                    name_found = True

                    for target_dim in override_info.keys():
                        override_value = override_info.get(target_dim)
                        if type(override_value) != int:
                            raise ValueError(f'override value of dim {target_dim} must be int')

                        if type(target_dim) == int and 0 <= target_dim < len(t.type.tensor_type.shape.dim):
                            # it is the index of the dim
                            t.type.tensor_type.shape.dim[target_dim].dim_value = override_value
                        elif type(target_dim) == str:
                            # it is the name of the dim
                            for dim in t.type.tensor_type.shape.dim:
                                if dim.WhichOneof('value') == 'dim_param' and dim.dim_param == target_dim:
                                    dim.dim_value = override_value
                                    break
                        else:
                            raise ValueError(f'wrong dim to be overrided: {target_dim}')

                    break
            if not name_found:
                raise ValueError(f'input tensor "{name}" is not found!')

    with open(output_path, mode='wb') as f:
        f.write(onnx_model.SerializeToString())

    # onnx_model = shape_inference.infer_shapes(onnx_model)
    # onnx_graph = onnx_model.graph
    return onnx_model

convert_dynamic("/home/gem/development/learn/wenet_stream_model_1201/model_8_4_fix_no_off/decoder_mnn.onnx",
                "/home/gem/development/learn/wenet_stream_model_1201/model_8_4_fix_no_off/decoder_fixed.onnx",
                {
                    "hyps": {"L": 11},
                    "r_hyps": {"L": 11},
                    "hyps_lens": {},
                    "encoder_out": {"T": 88}
                })
