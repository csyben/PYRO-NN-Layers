package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")


exports_files(["LICENSE"])

tf_custom_op_library(
    name = "pyronn_layers.so",
    srcs = glob(
        ["*.cc", "*.h", "**/*.h"],
        exclude = [
            "*.cu.cc",            
        ],
    ),
    gpu_srcs = glob(["*.cu.cc","*.h", "**/*.h"]),     
)

py_library(
    name = "cuda_operator",
    srcs = ["cuda_operator.py"],
    data = [":pyronn_layers.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

''' filegroup(
    name = "pyronn_layers",
    srcs = glob(
        ["*.cc"]
    ),
    gpu_srcs = glob(["*.cu"]),    
) '''

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__internal__"],
)
