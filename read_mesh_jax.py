import os
import jax
import pytest
import jax.numpy as jnp
from mpi4py import MPI
from mpi4jax import send, recv, bcast
from jax import random
from array_interfaces import array_factory, JaxStyleNumpyArray
import numpy as np
@pytest.mark.parametrize("backend", ["numpy", "numpy-immutable", "jax"])
@pytest.mark.parametrize("data, slice_idx, set_value", [
    ([1, 2, 3, 4, 5], slice(1, 4), 10),
    ([1.0, 2.0, 3.0, 4.0, 5.0], slice(0, 3), 20.0),
    ([1, 2, 3, 4, 5], slice(2, 5), 30),
    ([1.0, 2.0, 3.0, 4.0, 5.0], slice(1, 4), 40.0)
])

def test_array_slicing_and_setting(backend, data, slice_idx, set_value):
    arr = array_factory(data, backend=backend)
    if backend == "jax":
        arr = arr.at[slice_idx].set(set_value)
        assert jnp.all(arr[slice_idx] == set_value)
    elif backend == "numpy-immutable":
        arr = arr.at[slice_idx].set(set_value)
        assert np.all(arr[slice_idx] == set_value)
    else:
        arr.at[slice_idx].set(set_value)
        assert np.all(arr[slice_idx] == set_value)

use_jax=True
def set_item(arr, index, value):
    if use_jax:
        return arr.at[index].set(value)
    else:
        arr[index] = value
        return arr

def add_item(arr, index, value):
    if use_jax:
        return arr.at[index].add(value)
    else:
        arr[index] += value
        return arr
backend="jax"
# Set up environment variables for OpenMP
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MPI4JAX_USE_CUDA_MPI"] = "1"

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

# Ensure we have exactly 4 MPI tasks
assert mpi_size == 4, "This example requires exactly 4 MPI tasks."

# Ensure JAX uses the GPU
jax.config.update('jax_platform_name', 'cpu')

meshpath = '/home/dsidoren/myapps/test/pi/'
dist_mesh_dir = meshpath + 'dist_' + str(mpi_size) + '/'
file_name = dist_mesh_dir.strip() + '/rpart.out'

# Initialize variables
npes = jnp.zeros(1, dtype=jnp.int32)  # Initialize as JAX array

if rank == 0:
    with open(file_name, 'r') as file:
        # Read the number of processors
        #npes = jnp.array([int(file.readline().strip())], dtype=jnp.int32)
        npes=jnp.int32(int(file.readline().strip()))
        # Allocate partit%part array
        part = array_factory(np.zeros(npes+1, dtype=int), backend=backend)
        part = part.at[0].set(1)
        
        # Read the remaining integers into part(2:npes+1)
        remaining_integers = list(map(int, file.readline().strip().split()))
        part = part.at[slice(1, npes+ 1)].set(jnp.array(remaining_integers, dtype=jnp.int32))

        # Accumulate the part array
        for i in range(1, npes + 1):
            part = part.at[i].add(part[i - 1])

# Broadcast npes to all processes
npes = bcast(npes, root=0, comm=comm)

# Initialize part on other ranks
if rank != 0:
    part = jnp.zeros(npes[0] + 1, dtype=jnp.int32)

# Broadcast the part array to all processes
part = bcast(part, root=0, comm=comm)

# Print the part array in each process for debugging
print(f"Process {rank}: part = {part}")

# Finalize MPI
comm.Barrier()
MPI.Finalize()

