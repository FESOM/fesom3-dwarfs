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

force_rotation = True


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
        # npes = jnp.array([int(file.readline().strip())], dtype=jnp.int32)
        npes = jnp.int32(int(file.readline().strip()))
        # Allocate partit%part array
        part = jnp.zeros(npes + 1, dtype=jnp.int32)
        part = part.at[0].set(1)

        # Read the remaining integers into part(2:npes+1)
        remaining_integers = list(map(int, file.readline().strip().split()))
        part = part.at[slice(1, npes + 1)].set(jnp.array(remaining_integers, dtype=jnp.int32))
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
# print(f"Process {rank}: part = {part}")
##############################################################################
# READ MESH PARTITIONING
##############################################################################
file_name = dist_mesh_dir.strip() + 'my_list' + str(rank).zfill(5) + '.out'
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
chunk_size = 10000
mapping = jnp.zeros(chunk_size, dtype=jnp.int32)
ibuff = np.zeros((chunk_size, 4), dtype=np.int32)
rbuff = np.zeros((chunk_size, 3), dtype=np.float64)
mesh_check = 0
file_name = meshpath.strip() + '/nod2d.out'
with open(file_name, 'r') as file:
    if (rank == 0):
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
                iofs = partit_myList_nod2D[n] - nchunk * chunk_size - 1
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
            x = rbuff[n, 0] * np.pi / 180.
            y = rbuff[n, 1] * np.pi / 180.

            if mapping[n] >= 0:
                mesh_check += 1
                mesh_coord_nod2D = mesh_coord_nod2D.at[0, mapping[n]].set(x)
                mesh_coord_nod2D = mesh_coord_nod2D.at[1, mapping[n]].set(y)

# mesh_check_total = comm.allreduce(mesh_check, op=MPI.SUM)
print(mesh_check - partit_myDim_nod2D - partit_eDim_nod2D)
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
            ibuff[n, 0] = int(line[0]) - 1
            ibuff[n, 1] = int(line[1]) - 1
            ibuff[n, 2] = int(line[2]) - 1

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
                    mesh_elem2D = mesh_elem2D.at[m, n].set(-mapping[iofs] - 1)

    mesh_elem2D = -mesh_elem2D - 1

print(mesh_elem2D.min(), mesh_elem2D.max())
if rank == 0:
    print("elements are read")

# Read 3D auxiliary data
file_name = meshpath.strip() + '/aux3d.out'
mesh_nl = jnp.zeros(1, dtype=jnp.int32)
with open(file_name, 'r') as file:
    if rank == 0:
        mesh_nl = int(file.readline().strip())  # Read the number of levels
    # Broadcast the number of levels to all processes
    mesh_nl = comm.bcast(mesh_nl, root=0)

    # Check if the number of levels is less than 3
    if mesh_nl < 3:
        if rank == 0:
            print("!!!Number of levels is less than 3, model will stop!!!")
        MPI.COMM_WORLD.Abort(1)  # Stop execution

    # Allocate the array for storing the standard depths
    mesh_zbar = jnp.zeros(mesh_nl, dtype=jnp.float32)

    # Read the standard depths on the root process
    if rank == 0:
        file.readline()  # Skip the first line (already read)
        mesh_zbar = jnp.array([float(val) for val in file.readline().strip().split()])

    # Broadcast the zbar array to all processes
    mesh_zbar = comm.bcast(mesh_zbar, root=0)

    # Ensure zbar is negative
    if mesh_zbar[1] > 0:
        mesh_zbar = -mesh_zbar

    # Allocate the array for mid-depths of cells
    mesh_Z = 0.5 * (mesh_zbar[:-1] + mesh_zbar[1:])

    # Allocate the array for depth information
    mesh_depth = jnp.zeros(partit_myDim_nod2D + partit_eDim_nod2D, dtype=jnp.float32)

    # Initialize mesh_check
    mesh_check = 0

    # Process the data in chunks
    for nchunk in range((mesh_nod2D - 1) // chunk_size + 1):
        mapping = jnp.zeros(chunk_size, dtype=jnp.int32)

    # Create the mapping for the current chunk
    for n in range(partit_myDim_nod2D + partit_eDim_nod2D):
        ipos = (partit_myList_nod2D[n] - 1) // chunk_size
        if ipos == nchunk:
            iofs = partit_myList_nod2D[n] - nchunk * chunk_size - 1
            mapping = mapping.at[iofs].set(n + 1)  # Using 1-based indexing similar to Fortran

    # Determine the number of elements to read in this chunk
    k = min(chunk_size, mesh_nod2D - nchunk * chunk_size)

    # Read the depth values into the buffer on the root process
    if rank == 0:
        rbuff = np.zeros((k, 1), dtype=np.float32)
        for n in range(k):
            rbuff[n, 0] = float(file.readline().strip())

        # Broadcast the buffer to all processes
    rbuff = comm.bcast(rbuff, root=0)
    count = 0
    # Process the depths
    for n in range(k):
        x = rbuff[n, 0]
        if x > 0:
            x = -x  # Depths must be negative
        if x > -20.:  # Adjust based on threshold
            x = -20.
        if mapping[n] > 0:
            mesh_check += 1
            mesh_depth = mesh_depth.at[mapping[n] - 1].set(x)  # Adjust back to 0-based indexing
            count = count + 1

# ==============================
# Communication information
# Every process reads its file
# ==============================
MAX_NEIGHBOR_PARTITIONS = 10


class CommunicationData:
    def __init__(self):
        self.rPEnum = None
        self.rPE = None
        self.rptr = None
        self.rlist = None
        self.sPEnum = None
        self.sPE = None
        self.sptr = None
        self.slist = None


com_nod2D = CommunicationData()
com_elem2D = CommunicationData()
com_elem2D_full = CommunicationData()
max_neighbor_partitions = 10
# Set file paths
file_name = f"{dist_mesh_dir.strip()}/com_info{str(rank).zfill(5)}.out"
with open(file_name, 'r') as file:
    # Read the number of nodes
    n = int(file.readline().strip())

    # Read and validate rPEnum for nodes
    com_nod2D_rPEnum = int(file.readline().strip())
    if com_nod2D_rPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read rPE
    com_nod2D_rPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_nod2D_rPEnum])

    # Read rptr
    com_nod2D_rptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_nod2D_rPEnum + 1])

    # Allocate and read rlist
    com_nod2D_rlist = jnp.zeros(partit_eDim_nod2D, dtype=jnp.int32)
    count = 0
    while count < partit_eDim_nod2D:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_nod2D_rlist = com_nod2D_rlist.at[count:count + length].set(jnp.array(values))
        count += length

    # Read and validate sPEnum for nodes
    com_nod2D_sPEnum = int(file.readline().strip())
    if com_nod2D_sPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read sPE
    com_nod2D_sPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_nod2D_sPEnum])

    # Read sptr
    com_nod2D_sptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_nod2D_sPEnum + 1])

    # Allocate and read slist
    n_slist = com_nod2D_sptr[-1] - 1
    com_nod2D_slist = jnp.zeros(n_slist, dtype=jnp.int32)
    count = 0
    while count < n_slist:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_nod2D_slist = com_nod2D_slist.at[count:count + length].set(jnp.array(values))
        count += length

    # Read and validate rPEnum for elements
    com_elem2D_rPEnum = int(file.readline().strip())
    if com_elem2D_rPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read rPE
    com_elem2D_rPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_rPEnum])

    # Read rptr
    com_elem2D_rptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_rPEnum + 1])

    # Allocate and read rlist
    com_elem2D_rlist = jnp.zeros(partit_eDim_elem2D, dtype=jnp.int32)
    count = 0
    while count < partit_eDim_elem2D:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_elem2D_rlist = com_elem2D_rlist.at[count:count + length].set(jnp.array(values))
        count += length

    # Read and validate sPEnum for elements
    com_elem2D_sPEnum = int(file.readline().strip())
    if com_elem2D_sPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read sPE
    com_elem2D_sPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_sPEnum])

    # Read sptr
    com_elem2D_sptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_sPEnum + 1])

    # Allocate and read slist
    n_slist = com_elem2D_sptr[-1] - 1
    com_elem2D_slist = jnp.zeros(n_slist, dtype=jnp.int32)
    count = 0
    while count < n_slist:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_elem2D_slist = com_elem2D_slist.at[count:count + length].set(jnp.array(values))
        count += length

    # Read and validate rPEnum for full elements
    com_elem2D_full_rPEnum = int(file.readline().strip())
    if com_elem2D_full_rPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read rPE
    com_elem2D_full_rPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_full_rPEnum])

    # Read rptr
    com_elem2D_full_rptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_full_rPEnum + 1])

    # Allocate and read rlist
    com_elem2D_full_rlist = jnp.zeros(partit_eDim_elem2D + partit_eXDim_elem2D, dtype=jnp.int32)
    count = 0
    while count < partit_eDim_elem2D + partit_eXDim_elem2D:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_elem2D_full_rlist = com_elem2D_full_rlist.at[count:count + length].set(jnp.array(values))
        count += length

    # Read and validate sPEnum for full elements
    com_elem2D_full_sPEnum = int(file.readline().strip())
    if com_elem2D_full_sPEnum > max_neighbor_partitions:
        raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

    # Read sPE
    com_elem2D_full_sPE = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_full_sPEnum])

    # Read sptr
    com_elem2D_full_sptr = jnp.array(list(map(int, file.readline().strip().split()))[:com_elem2D_full_sPEnum + 1])

    # Allocate and read slist
    n_slist = com_elem2D_full_sptr[-1] - 1
    com_elem2D_full_slist = jnp.zeros(n_slist, dtype=jnp.int32)
    count = 0
    while count < n_slist:
        values = list(map(int, file.readline().strip().split()))
        length = len(values)
        com_elem2D_full_slist = com_elem2D_full_slist.at[count:count + length].set(jnp.array(values))
        count += length

if rank == 0:
    print("Communication arrays are read")

del rbuff
del ibuff
del mapping
print(rank, partit_eDim_nod2D, com_nod2D_rlist)
comm.Barrier()
MPI.Finalize()