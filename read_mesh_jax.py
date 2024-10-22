import os
import jax
import pytest
import jax.numpy as jnp
from mpi4py import MPI
from mpi4jax import send, recv, bcast
from jax import random
from jax import ops
from array_interfaces import array_factory, JaxStyleNumpyArray
import numpy as np
from data_types import Mesh, Partitioning
force_rotation = True


# Set up environment variables for OpenMP
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MPI4JAX_USE_CUDA_MPI"] = "1"

# Ensure JAX uses the GPU
jax.config.update('jax_platform_name', 'cpu')

mesh = Mesh()
# Initialize MPI
# comm = MPI.COMM_WORLD
partit = Partitioning(npes=MPI.COMM_WORLD.Get_size(), mype=MPI.COMM_WORLD.Get_rank(), MPI_COMM_FESOM=MPI.COMM_WORLD)
# Ensure we have exactly 4 MPI tasks
assert partit.npes == 4, "This example requires exactly 4 MPI tasks."
meshpath = '/home/dsidoren/myapps/test/pi/'
from read_mesh_and_partition import *

mesh, partit=read_mesh_and_partition(mesh, partit, meshpath)
cyclic_length=2.*jnp.pi
#test exchange:
arr = jnp.ones(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
arr = arr.at[partit.myDim_nod2D:].set(0)
print("before exchange:", partit.mype, min(arr), max(arr))
arr=exchange_nod2D_i(arr, partit)
print("after exchange:", partit.mype, min(arr), max(arr))

mesh, partit=test_tri(mesh, partit, cyclic_length)
mesh, partit=load_edges(mesh, partit, meshpath)
mesh, partit=find_neighbors(mesh, partit)
partit.MPI_COMM_FESOM.Barrier()
MPI.Finalize()