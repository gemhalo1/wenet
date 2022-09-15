import tvm
from tvm import relay, rpc, runtime, auto_scheduler
from tvm.auto_scheduler.utils import request_remote
from onnx_model import OnnxModel
import os
import numpy as np


def compile_onnx_model(model_path, target, override_dynamic_inputs=None, use_ndk=False):
    onnx_model = OnnxModel(model_path, override_dynamic_inputs=override_dynamic_inputs)

    inputs_ = onnx_model.inputs

    shape_dict = dict(zip(onnx_model.inputs, [onnx_model.tensor_map[x].tensor_type.shape for x in onnx_model.inputs]))
    mod, params = relay.frontend.from_onnx(onnx_model.onnx_model, shape_dict)
    hardware_params = None

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, hardware_params=hardware_params)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    log_file = "/tmp/log_%s.json" % (target.kind.name)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance

        # if use ndk, don't forget to set environment variable TVM_NDK_CC=/home/gem/Android/Sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.LocalRunner(
            timeout=30,
            repeat=10,
            min_repeat_ms=200,
            enable_cpu_cache_flush=True,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            return lib


local_cpu_target = tvm.target.Target(target="llvm", host="llvm")

encoder_lib = compile_onnx_model('/home/gem/development/learn/wenet_stream_model_0830/model_16_4/a.onnx', # '/home/gem/development/learn/wenet_stream_model_0830/no_offset/encoder.onnx',
                                 local_cpu_target,
                                 override_dynamic_inputs=None,
                                 use_ndk=False
                                )

print(encoder_lib)

# if the filename is 'xxx.so', then a big so is exported, how does it work?
# if 'xxx.tar', then it contains two files: lib0.o, devc.o, how does it work?
# how to export regular .so, .params, .json files?
encoder_lib.export_library('encoder.tar')
