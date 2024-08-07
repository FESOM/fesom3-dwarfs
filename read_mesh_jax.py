import os
import jax
import jax.numpy as jnp
from mpi4py import MPI
from mpi4jax import send, recv, bcast
from jax import random
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
        part = jnp.zeros(npes + 1, dtype=jnp.int32)
        part=set_item(part,0,1)
        
        # Read the remaining integers into part(2:npes+1)
        remaining_integers = list(map(int, file.readline().strip().split()))
        part = set_item(part, slice(1, npes+ 1), jnp.array(remaining_integers, dtype=jnp.int32))

        # Accumulate the part array
        for i in range(1, npes + 1):
            part = add_item(part, i, part[i - 1])

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

