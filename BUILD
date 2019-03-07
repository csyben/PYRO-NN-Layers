# Description:
# Code examples referenced by adding_an_op

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cuda_tests_tags")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")


exports_files(["LICENSE"])

tf_custom_op_library(
    name = "lme_custom_ops.so",
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
    data = [":lme_custom_ops.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

''' filegroup(
    name = "lme_custom_ops",
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
