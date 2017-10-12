licenses(["restricted"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

load("//third_party/hdf5:hdf5.bzl", "hdf5_hdr")
load("//third_party/hdf5:hdf5.bzl", "hdf5_libs")
load("//third_party/hdf5:hdf5.bzl", "hdf5_flags")
load("//third_party/hdf5:hdf5.bzl", "if_hdf5")

cc_library(
    name = "hdf5",
    srcs = if_hdf5(hdf5_libs()),
    hdrs = if_hdf5(glob(hdf5_hdr())),
    copts = if_hdf5(hdf5_flags()),
    visibility = ["//visibility:public"],
    deps = [
        "@zlib_archive//:zlib",
    ]
)
