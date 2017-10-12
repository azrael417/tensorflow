#OpenMPI and Mvapich/mpich require different headers
#based on the configuration options return one or the other

def mpi_hdr():
    MPI_LIB_IS_OPENMPI=False
    hdrs = [ "*.h" ]
    return hdrs

def mpi_lib():
    MPI_LIB_IS_OPENMPI=False
    lib=None
    if MPI_LIB_IS_OPENMPI:
        lib="libmpi.so"   							#When using OpenMPI
    else:
        lib="libmpich.so"        #When using MVAPICH
    return [lib]

def if_mpi(if_true, if_false = []):
    return select({
        "//tensorflow:with_mpi_support": if_true,
        "//conditions:default": if_false
    })