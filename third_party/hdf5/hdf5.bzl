#Instructions for building with HDF5 support
#based on the configuration options return one or the other

def if_hdf5(if_true, if_false = []):
    return select({
        "//tensorflow:with_hdf5_support": if_true,
        "//conditions:default": if_false
    })
    
def hdf5_hdr():
    hdrs = [ "hdf5.h" ]
    return hdrs

def hdf5_flags():
    flags = ["TENSORFLOW_USE_HDF5"]
    return flags

def hdf5_libs():
    libs = ["libhdf5.so","libz.so"]
    return libs
