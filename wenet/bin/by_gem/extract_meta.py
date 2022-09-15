import onnx
import json

def extract_meta(onnx_file, json_file):
    model = onnx.load(onnx_file)

    meta = {x.key:x.value for x in model.metadata_props}

    with open(json_file, mode='w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    extract_meta('encoder.onnx', 'meta.json')

# cmake -DMNN_DEBUG_TENSOR_SIZE=ON -DONNX=ON -DMNN=ON -DTORCH=OFF -DWEBSOCET=ON -DGRPC=OFF
