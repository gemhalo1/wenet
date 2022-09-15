import typing
from collections import namedtuple

import numpy as np
import onnx
from onnx import shape_inference
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE, STORAGE_TENSOR_TYPE_TO_FIELD
from onnx.onnx_ml_pb2 import TensorProto, TypeProto, ValueInfoProto, AttributeProto, NodeProto


class OnnxModel(object):
    TensorType = namedtuple("TensorType", ["dtype", "shape"])
    Node = namedtuple("Node", ["name", "op", "inputs", "outputs", "attrs"])
    Tensor = namedtuple("Tensor", ["name", "tensor_type", "value"], defaults=[None, None, None])

    def __init__(self, onnx_path, override_dynamic_inputs=None):
        self.onnx_model = shape_inference.infer_shapes(onnx.load(onnx_path), check_type=True, strict_mode=True, data_prop=True)
        self.onnx_graph = self.onnx_model.graph

        if override_dynamic_inputs is not None:
            for name in override_dynamic_inputs.keys():
                override_info = override_dynamic_inputs.get(name)

                name_found = False
                for t in self.onnx_graph.input:
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

        self.onnx_model = shape_inference.infer_shapes(self.onnx_model)
        self.onnx_graph = self.onnx_model.graph

        self.parse()

    def convert_tensor_type(self, t: TypeProto) -> typing.Union[TensorType, None]:
        if t.WhichOneof('value') == "tensor_type":
            dtype = TensorProto.DataType.Name(t.tensor_type.elem_type)
            shape = []

            if t.tensor_type.HasField('shape'):
                shape = []
                if len(t.tensor_type.shape.dim):
                    for d in t.tensor_type.shape.dim:
                        if d.WhichOneof('value') == "dim_value":
                            shape.append(d.dim_value)
                        else:  # should be string dim_param
                            shape.append(None)
                else:
                    # it's a scalar
                    pass

            return OnnxModel.TensorType(dtype=dtype, shape=tuple(shape))

        # not expected type
        return None

    def convert_value_info(self, v: ValueInfoProto) -> Tensor:
        """
        convert ValueInfoProto to ValueInfo
        :param v:
        :return:
        """

        return OnnxModel.Tensor(name=v.name, tensor_type=self.convert_tensor_type(v.type), value=None)

    def convert_tensor(self, t: TensorProto):
        shape = tuple(t.dims)
        dtype = TensorProto.DataType.Name(t.data_type)

        # data_location should be either 'DEFAULT' or 'EXTERNAL', but currently only supports 'DEFAULT'
        data_location = TensorProto.DataLocation.Name(t.data_location)
        assert data_location == 'DEFAULT'

        np_dtype = TENSOR_TYPE_TO_NP_TYPE[t.data_type]
        if t.raw_data:
            value = np.frombuffer(t.raw_data, dtype=np_dtype).reshape(shape)
        else:
            # should be one of float_data, int32_data, string_data, int64_data,
            attr_name = STORAGE_TENSOR_TYPE_TO_FIELD[t.data_type]
            value = np.array(t.__getattribute__(attr_name), dtype=np_dtype)
            #TODO: shall handle COMPLEX64 and COMPLEX128 types

        return OnnxModel.Tensor(name=t.name, tensor_type=OnnxModel.TensorType(dtype=dtype, shape=shape), value=value)

    def convert_attribute(self, attr: AttributeProto):
        value = None

        if attr.type == AttributeProto.FLOAT:
            value = attr.f
        elif attr.type == AttributeProto.INT:
            value = attr.i
        elif attr.type == AttributeProto.STRING:
            value = attr.s.decode('utf-8')
        elif attr.type == AttributeProto.TENSOR:
            value = self.convert_tensor(attr.t)
        elif attr.type == AttributeProto.GRAPH:
            pass
        elif attr.type == AttributeProto.SPARSE_TENSOR:
            pass
        elif attr.type == AttributeProto.TYPE_PROTO:
            pass
        elif attr.type == AttributeProto.FLOATS:
            value = attr.floats
        elif attr.type == AttributeProto.INTS:
            value = attr.ints
        elif attr.type == AttributeProto.STRINGS:
            value = attr.strings
        elif attr.type == AttributeProto.TENSORS:
            pass
        elif attr.type == AttributeProto.GRAPHS:
            pass
        elif attr.type == AttributeProto.SPARSE_TENSORS:
            pass
        elif attr.type == AttributeProto.TYPE_PROTOS:
            pass

        return attr.name, value

    def convert_node(self, node: NodeProto):
        attrs = dict([self.convert_attribute(attr) for attr in node.attribute])
        return OnnxModel.Node(name=node.name, op=node.op_type, inputs=list(node.input), outputs=list(node.output), attrs=attrs)

    def parse(self):
        # graph.input, graph.output, graph.value_info are repeated ValueInfoProto
        inputs = [self.convert_value_info(x) for x in self.onnx_graph.input]
        outputs = [self.convert_value_info(x) for x in self.onnx_graph.output]

        # 中间结果tensor，没有权重值
        value_info = [self.convert_value_info(x) for x in self.onnx_graph.value_info]

        # 权重tensor
        constant_tensors = [self.convert_tensor(t) for t in self.onnx_graph.initializer]

        self.all_tensors = inputs + value_info + constant_tensors + outputs

        self.tensor_map = {}
        for t in self.all_tensors:
            self.tensor_map[t.name] = t

        self.inputs = [x.name for x in inputs]
        self.outputs = [x.name for x in outputs]

        self.nodes = [self.convert_node(n) for n in self.onnx_graph.node]

        # ??? 如何处理量化  quantization_annotation

        # print(self.onnx_model.ir_version, self.onnx_model.model_version, [x.version for x in self.onnx_model.opset_import])

    def is_constant_tensor(self, name):
        tensor = self.tensor_map.get(name)
        return tensor is not None and tensor.value is not None

    def nodes_with_inputs(self, inputs):
        result_nodes = []

        inputs_set = set(inputs)
        for node in self.nodes:
            if set(node.inputs).intersection(inputs_set):
                result_nodes.append(node)

        return result_nodes

    def get_subgraph(self, outputs: typing.List[str], inputs: typing.Union[typing.List[str], None]=None):
        # make a map from tensor name to its producer node
        producer_map = {}
        for n in self.onnx_graph.node:
            for o in n.output:
                producer_map[o] = n

        used_tensors = set()
        input_tensors = set()
        output_tensors = set()
        used_nodes = {}

        worklist = list(outputs)

        while len(worklist) > 0:
            o = worklist.pop(0)

            if o in used_tensors:
                continue

            if self._find_input(o) is not None:
                input_tensors.add(o)
                continue

            if self.find_constant(o) is not None:
                used_tensors.add(o)
                continue

            tensor = self._find_output(o)
            if tensor is not None:
                if inputs is not None and o in inputs:
                    input_tensors.add(o)
                else:
                    output_tensors.add(o)
            else:
                tensor = self._find_intermediate(o)
                assert tensor is not None

                if o in outputs:
                    output_tensors.add(o)
                elif inputs is not None and o in inputs:
                    input_tensors.add(o)
                else:
                    used_tensors.add(o)

            producer = producer_map.get(tensor.name)
            used_nodes[producer.name] = producer

            worklist.extend(producer.input)

        for o in list(input_tensors):
            producer = producer_map.get(o)
            if producer is not None and producer.name in used_nodes:
                input_tensors.remove(o)

        # print('inputs: ', input_tensors)
        # print('outputs: ', output_tensors)
        # print('used_tensors: ', used_tensors)
        # print('nodes: ', [n.op_type + ' : ' + n.name for n in used_nodes.values()])

    def _find_by_name(self, theList, name):
        for x in theList:
            if x.name == name:
                return x
        return None

    def _find_input(self, name):
        return self._find_by_name(self.onnx_graph.input, name)

    def _find_output(self, name):
        return self._find_by_name(self.onnx_graph.output, name)

    def _find_intermediate(self, name):
        return self._find_by_name(self.onnx_graph.value_info, name)

    def find_constant(self, name):
        return self._find_by_name(self.onnx_graph.initializer, name)
