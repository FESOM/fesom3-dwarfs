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


force_rotation=True
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
        part = jnp.zeros(npes+1, dtype=jnp.int32)
        part = part.at[0].set(1)
        
        # Read the remaining integers into part(2:npes+1)
        remaining_integers = list(map(int, file.readline().strip().split()))
        part = part.at[slice(1, npes+ 1)].set(jnp.array(remaining_integers, dtype=jnp.int32))
        # Accumulate the part array
        for i in range(1, npes + 1):
            part = part.at[i].add(part[i - 1])

# Broadcast npes to all processes
npes = comm.bcast(npes, root=0)

# Initialize part on other ranks
if rank != 0:
    part = jnp.zeros(npes + 1, dtype=jnp.int32)
# Broadcast the part array to all processes
part = comm.bcast(part, root=0)
# Print the part array in each process for debugging
#print(f"Process {rank}: part = {part}")
##############################################################################
# READ MESH PARTITIONING
##############################################################################
file_name = dist_mesh_dir.strip() + 'my_list'+str(rank).zfill(5) + '.out'
with open(file_name, 'r') as file:
    # Read the value of n
    n = int(file.readline().strip())

    # Read partit%myDim_nod2D
    partit_myDim_nod2D = jnp.int32(file.readline().strip())

    # Read partit%eDim_nod2D
    partit_eDim_nod2D = jnp.int32(file.readline().strip())

    # Allocate partit%myList_nod2D
    partit_myList_nod2D = jnp.zeros(partit_myDim_nod2D + partit_eDim_nod2D, dtype=jnp.int32)

    count = 0
    while count < partit_myList_nod2D.size:
        line = file.readline().strip()  # Assuming 'file' is already an open file object
        numbers = map(int, line.split())
        for num in numbers:
            if count < partit_myList_nod2D.size:
                partit_myList_nod2D = partit_myList_nod2D.at[count].set(num)
                count += 1
            else:
                break

    # Read partit%myDim_elem2D
    partit_myDim_elem2D = jnp.int32(file.readline().strip())

    # Read partit%eDim_elem2D
    partit_eDim_elem2D = jnp.int32(file.readline().strip())

    # Read partit%eXDim_elem2D
    partit_eXDim_elem2D = jnp.int32(file.readline().strip())

    # Allocate partit%myList_elem2D
    partit_myList_elem2D = jnp.zeros(partit_myDim_elem2D + partit_eDim_elem2D + partit_eXDim_elem2D, dtype=jnp.int32)

    # Read partit%myList_elem2D
    count = 0
    while count < partit_myList_elem2D.size:
        line = file.readline().strip()  # Assuming 'file' is already an open file object
        numbers = map(int, line.split())
        for num in numbers:
            if count < partit_myList_elem2D.size:
                partit_myList_elem2D = partit_myList_elem2D.at[count].set(num)
                count += 1
            else:
                break

    # Read partit%myDim_edge2D
    partit_myDim_edge2D = jnp.int32(file.readline().strip())

    # Read partit%eDim_edge2D
    partit_eDim_edge2D = jnp.int32(file.readline().strip())

    # Allocate partit%myList_edge2D
    partit_myList_edge2D = jnp.zeros(partit_myDim_edge2D + partit_eDim_edge2D, dtype=jnp.int32)

    # Read partit%myList_edge2D
    count = 0
    while count < partit_myList_edge2D.size:
        line = file.readline().strip()  # Assuming 'file' is already an open file object
        numbers = map(int, line.split())
        for num in numbers:
            if count < partit_myList_edge2D.size:
                partit_myList_edge2D = partit_myList_edge2D.at[count].set(num)
                count += 1
            else:
                break
##############################################################################
# Read 2D node data
mesh_nod2D = part[npes] - 1
# Allocate mesh_coord_nod2D with JAX
mesh_coord_nod2D = jnp.zeros((2, partit_myDim_nod2D + partit_eDim_nod2D), dtype=jnp.float32)
error_status = 0
# like in Fortran we read the mesh in chunks to avoid loading large arrays into memory
chunk_size   = 10000
mapping = jnp.zeros(chunk_size,     dtype=jnp.int32)
ibuff   = np.zeros((chunk_size, 4), dtype=np.int32)
rbuff   = np.zeros((chunk_size, 3), dtype=np.float64)
mesh_check=0
file_name = meshpath.strip() + '/nod2d.out'
with open(file_name, 'r') as file:
    if (rank==0):
        n = int(file.readline().strip())  # Read nod2D
        if n != mesh_nod2D:
            error_status = 1  # Set the error status for consistency between part and nod2D
        print('reading', file_name)

    # Broadcast error status to all processes
    error_status = MPI.COMM_WORLD.bcast(error_status, root=0)

    if error_status != 0:
        print(n)
        print('error: nod2D != part[npes]')
        MPI.COMM_WORLD.Abort(1)  # Stop execution if there's an error
    for nchunk in range((mesh_nod2D - 1) // chunk_size + 1):
        # Create the mapping for the current chunk
        mapping = mapping.at[:chunk_size].set(-1)
        for n in range(partit_myDim_nod2D + partit_eDim_nod2D):
            ipos = (partit_myList_nod2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit_myList_nod2D[n] - nchunk * chunk_size-1
                mapping = mapping.at[iofs].set(n)

        # Read the chunk into the buffers
        k = min(chunk_size, mesh_nod2D - nchunk * chunk_size)
        if rank == 0:
            for n in range(k):
                line = file.readline().strip().split()
                ibuff[n, 0], rbuff[n, 0], rbuff[n, 1], ibuff[n, 1] = int(line[0]), float(line[1]), float(line[2]), int(
                line[3])

                # Apply the offset for longitude shift
                offset = 0.0
                if rbuff[n, 0] > 180.0:
                    offset = -360.0
                elif rbuff[n, 0] < -180.0:
                    offset = 360.0

        # Broadcast the buffers
        rbuff[:, 0] = comm.bcast(rbuff[:, 0], root=0)
        rbuff[:, 1] = comm.bcast(rbuff[:, 1], root=0)
        ibuff[:, 1] = comm.bcast(ibuff[:, 1], root=0)

        # Fill the local arrays
        for n in range(k):
            x = rbuff[n, 0] * np.pi/180.
            y = rbuff[n, 1] * np.pi/180.

            if mapping[n] >= 0:
                mesh_check += 1
                mesh_coord_nod2D = mesh_coord_nod2D.at[0, mapping[n]].set(x)
                mesh_coord_nod2D = mesh_coord_nod2D.at[1, mapping[n]].set(y)

#mesh_check_total = comm.allreduce(mesh_check, op=MPI.SUM)
print(mesh_check-partit_myDim_nod2D - partit_eDim_nod2D)
##############################################################################
# Read 2D element data
mesh_elem2D = jnp.zeros((3, partit_myDim_elem2D + partit_eDim_elem2D + partit_eXDim_elem2D), dtype=jnp.int32)
file_name = meshpath.strip() + '/elem2d.out'
with open(file_name, 'r') as file:
    mesh_elem2D_total = jnp.int32(0)
    if rank == 0:
        mesh_elem2D_total = jnp.int32(file.readline().strip())  # Read the total number of elem2D
        print('reading', file_name)

    # Broadcast the total number of elem2D to all processes
    mesh_elem2D_total = comm.bcast(mesh_elem2D_total, root=0)

    # Loop over chunks and process the data
    for nchunk in range((mesh_elem2D_total - 1) // chunk_size + 1):
        mapping = mapping.at[:chunk_size].set(-1)

        for n in range(partit_myDim_elem2D + partit_eDim_elem2D + partit_eXDim_elem2D):
            ipos = (partit_myList_elem2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit_myList_elem2D[n] - nchunk * chunk_size - 1
                mapping = mapping.at[iofs].set(n)

    k = min(chunk_size, mesh_elem2D_total - nchunk * chunk_size)
    if rank == 0:
        for n in range(k):
            line = file.readline().strip().split()
            ibuff[n, 0]=int(line[0])-1
            ibuff[n, 1]=int(line[1])-1
            ibuff[n, 2]=int(line[2])-1

    # Broadcast the buffers
    ibuff[:, 0] = comm.bcast(ibuff[:, 0], root=0)
    ibuff[:, 1] = comm.bcast(ibuff[:, 1], root=0)
    ibuff[:, 2] = comm.bcast(ibuff[:, 2], root=0)

    # Fill the local arrays
    for n in range(k):
        if mapping[n] >= 0:
            mesh_elem2D = mesh_elem2D.at[0, mapping[n]].set(ibuff[n, 0])
            mesh_elem2D = mesh_elem2D.at[1, mapping[n]].set(ibuff[n, 1])
            mesh_elem2D = mesh_elem2D.at[2, mapping[n]].set(ibuff[n, 2])

    # Convert global to local numbering
    for nchunk in range((mesh_elem2D_total - 1) // chunk_size + 1):
        mapping = mapping.at[:chunk_size].set(0)

        for n in range(partit_myDim_nod2D + partit_eDim_nod2D):
            ipos = (partit_myList_nod2D[n]) // chunk_size
            if ipos == nchunk:
                iofs = partit_myList_nod2D[n] - nchunk * chunk_size - 1
                mapping = mapping.at[iofs].set(n)

        for n in range(partit_myDim_elem2D + partit_eDim_elem2D + partit_eXDim_elem2D):
            for m in range(3):
                nn = mesh_elem2D[m, n]
                ipos = (nn) // chunk_size
                if ipos == nchunk:
                    iofs = nn - nchunk * chunk_size - 1
                    mesh_elem2D = mesh_elem2D.at[m, n].set(-mapping[iofs]-1)

    mesh_elem2D = -mesh_elem2D-1

print(mesh_elem2D.min(), mesh_elem2D.max())
if rank == 0:
    print("elements are read")



#print(partit_myList_nod2D.min())
# Finalize MPI
comm.Barrier()
MPI.Finalize()

