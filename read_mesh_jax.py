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


set_mesh_transform_matrix(50.*jnp.pi/180., 15.*jnp.pi/180., -90.*jnp.pi/180.)

mesh, partit=read_mesh_and_partition(mesh, partit, meshpath, force_rotation=True)


print("mesh.elem2D=", mesh.elem2D.min(), mesh.elem2D.max())

cyclic_length=2.*jnp.pi
r_earth=6367500.0
#test exchange:
arr = jnp.ones(partit.myDim_elem2D + partit.eDim_elem2D, dtype=jnp.float32)
arr = arr.at[partit.myDim_elem2D:].set(0)
print("before exchange:", partit.mype, min(arr), max(arr))
arr=exchange_elem2D(arr, partit)
print("after exchange:", partit.mype, min(arr), max(arr))

mesh, partit=test_tri(mesh, partit, cyclic_length)

mesh, partit=load_edges(mesh, partit, meshpath)
mesh, partit=find_neighbors(mesh, partit)
mesh, partit=find_levels(mesh, partit, meshpath)
mesh, partit=find_levels_min_e2n(mesh, partit)

elnodes = mesh.elem2D[:, 0]
print("x coord check:", partit.mype, mesh.coord_nod2D[0, elnodes])
print("y coord check:", partit.mype, mesh.coord_nod2D[1, elnodes])

mesh, partit=mesh_areas(mesh, partit, cartesian=False, cyclic_length=cyclic_length, r_earth=r_earth)
partit.MPI_COMM_FESOM.Barrier()
MPI.Finalize()