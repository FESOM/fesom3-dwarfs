import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from mpi4jax import send, recv, bcast
from module_rotate_grid import *

def read_mesh_and_partition(mesh, partit, meshpath):
    dist_mesh_dir = meshpath + 'dist_' + str(partit.npes) + '/'
    file_name = dist_mesh_dir.strip() + '/rpart.out'

    # Initialize variables
    partit.npes = jnp.zeros(1, dtype=jnp.int32)  # Initialize as JAX array

    if partit.mype == 0:
        with open(file_name, 'r') as file:
            # Read the number of processors
            partit.npes = jnp.int32(int(file.readline().strip()))
            # Allocate partit%part array
            partit.part = jnp.zeros(partit.npes + 1, dtype=jnp.int32)
            partit.part = partit.part.at[0].set(1)
            # Read the remaining integers into part(2:npes+1)
            remaining_integers = list(map(int, file.readline().strip().split()))
            partit.part = partit.part.at[slice(1, partit.npes + 1)].set(jnp.array(remaining_integers, dtype=jnp.int32))
            # Accumulate the part array
            for i in range(1, partit.npes + 1):
                partit.part = partit.part.at[i].add(partit.part[i - 1])
    # Broadcast npes to all processes
    partit.npes = partit.MPI_COMM_FESOM.bcast(partit.npes, root=0)

    # Initialize part on other ranks
    if partit.mype != 0:
        partit.part = jnp.zeros(partit.npes + 1, dtype=jnp.int32)
    # Broadcast the part array         partit.part=partto all processes
    partit.part = partit.MPI_COMM_FESOM.bcast(partit.part, root=0)

    # Print the part array in each process for debugging
    # print(f"Process {rank}: part = {part}")
    ##############################################################################
    # READ MESH PARTITIONING
    ##############################################################################
    file_name = dist_mesh_dir.strip() + 'my_list' + str(partit.mype).zfill(5) + '.out'
    with open(file_name, 'r') as file:
        # Read the value of n
        n = int(file.readline().strip())

        # Read partit%myDim_nod2D
        partit.myDim_nod2D = jnp.int32(file.readline().strip())

        # Read partit%eDim_nod2D
        partit.eDim_nod2D = jnp.int32(file.readline().strip())

        # Allocate partit%myList_nod2D
        partit.myList_nod2D = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)

        count = 0
        while count < partit.myList_nod2D.size:
            line = file.readline().strip()  # Assuming 'file' is already an open file object
            numbers = map(int, line.split())
            for num in numbers:
                if count < partit.myList_nod2D.size:
                    partit.myList_nod2D = partit.myList_nod2D.at[count].set(num)
                    count += 1
                else:
                    break

        # Read partit%myDim_elem2D
        partit.myDim_elem2D = jnp.int32(file.readline().strip())

        # Read partit%eDim_elem2D
        partit.eDim_elem2D = jnp.int32(file.readline().strip())

        # Read partit%eXDim_elem2D
        partit.eXDim_elem2D = jnp.int32(file.readline().strip())

        # Allocate partit%myList_elem2D
        partit.myList_elem2D = jnp.zeros(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D,
                                         dtype=jnp.int32)

        # Read partit%myList_elem2D
        count = 0
        while count < partit.myList_elem2D.size:
            line = file.readline().strip()  # Assuming 'file' is already an open file object
            numbers = map(int, line.split())
            for num in numbers:
                if count < partit.myList_elem2D.size:
                    partit.myList_elem2D = partit.myList_elem2D.at[count].set(num)
                    count += 1
                else:
                    break
        # Read partit%myDim_edge2D
        partit.myDim_edge2D = jnp.int32(file.readline().strip())

        # Read partit%eDim_edge2D
        partit.eDim_edge2D = jnp.int32(file.readline().strip())

        # Allocate partit%myList_edge2D
        partit.myList_edge2D = jnp.zeros(partit.myDim_edge2D + partit.eDim_edge2D, dtype=jnp.int32)

        # Read partit%myList_edge2D
        count = 0
        while count < partit.myList_edge2D.size:
            line = file.readline().strip()  # Assuming 'file' is already an open file object
            numbers = map(int, line.split())
            for num in numbers:
                if count < partit.myList_edge2D.size:
                    partit.myList_edge2D = partit.myList_edge2D.at[count].set(num)
                    count += 1
                else:
                    break
    ##############################################################################
    # Read 2D node data
    mesh.nod2D = partit.part[partit.npes] - 1
    # Allocate mesh.coord_nod2D with JAX
    mesh.coord_nod2D = jnp.zeros((2, partit.myDim_nod2D + partit.eDim_nod2D), dtype=jnp.float32)
    error_status = 0
    # like in Fortran we read the mesh in chunks to avoid loading large arrays into memory
    chunk_size = 1000000
    mapping = jnp.zeros(chunk_size, dtype=jnp.int32)
    ibuff = np.zeros((chunk_size, 4), dtype=np.int32)
    rbuff = np.zeros((chunk_size, 3), dtype=np.float64)
    mesh.check = 0
    file_name = meshpath.strip() + '/nod2d.out'
    with open(file_name, 'r') as file:
        if (partit.mype == 0):
            n = int(file.readline().strip())  # Read nod2D
            if n != mesh.nod2D:
                error_status = 1  # Set the error status for consistency between part and nod2D
            print('reading', file_name)

        # Broadcast error status to all processes
        error_status = MPI.COMM_WORLD.bcast(error_status, root=0)

        if error_status != 0:
            print(n)
            print('error: mesh.nod2D != part[npes]', mesh.nod2D, n)
            MPI.COMM_WORLD.Abort(1)  # Stop execution if there's an error
        for nchunk in range((mesh.nod2D - 1) // chunk_size + 1):
            # Create the mapping for the current chunk
            mapping = mapping.at[:chunk_size].set(-1)
            for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
                ipos = (partit.myList_nod2D[n] - 1) // chunk_size
                if ipos == nchunk:
                    iofs = partit.myList_nod2D[n] - nchunk * chunk_size - 1
                    mapping = mapping.at[iofs].set(n)

            # Read the chunk into the buffers
            k = min(chunk_size, mesh.nod2D - nchunk * chunk_size)
            if partit.mype == 0:
                for n in range(k):
                    line = file.readline().strip().split()
                    ibuff[n, 0], rbuff[n, 0], rbuff[n, 1], ibuff[n, 1] = int(line[0]), float(line[1]), float(
                        line[2]), int(
                        line[3])

                    # Apply the offset for longitude shift
                    offset = 0.0
                    if rbuff[n, 0] > 180.0:
                        offset = -360.0
                    elif rbuff[n, 0] < -180.0:
                        offset = 360.0

            # Broadcast the buffers
            rbuff[:, 0] = partit.MPI_COMM_FESOM.bcast(rbuff[:, 0], root=0)
            rbuff[:, 1] = partit.MPI_COMM_FESOM.bcast(rbuff[:, 1], root=0)
            ibuff[:, 1] = partit.MPI_COMM_FESOM.bcast(ibuff[:, 1], root=0)

            # Fill the local arrays
            for n in range(k):
                x = rbuff[n, 0] * np.pi / 180.
                y = rbuff[n, 1] * np.pi / 180.

                if mapping[n] >= 0:
                    mesh.check += 1
                    mesh.coord_nod2D = mesh.coord_nod2D.at[0, mapping[n]].set(x)
                    mesh.coord_nod2D = mesh.coord_nod2D.at[1, mapping[n]].set(y)

    # mesh.check_total = partit.MPI_COMM_FESOM.allreduce(mesh.check, op=MPI.SUM)
    print(mesh.check - partit.myDim_nod2D - partit.eDim_nod2D)
    ##############################################################################

    # Read 2D element data
#   mesh.elem2D = jnp.zeros((3, partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D), dtype=jnp.int32)
    mesh.elem2D = jnp.full((3, partit.myDim_elem2D), -1, dtype=jnp.int32)
    file_name = meshpath.strip() + '/elem2d.out'
    with open(file_name, 'r') as file:
        mesh.elem2D_total = jnp.int32(0)
        mesh.elem2D_total = jnp.int32(file.readline().strip())  # Read the total number of elem2D
        print('reading', file_name)
        # Loop over chunks and process the data
        mapping = mapping.at[:mesh.elem2D_total].set(-1)
        for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
            ipos = partit.myList_elem2D[n] - 1
            mapping = mapping.at[ipos].set(n)
        for n in range(mesh.elem2D_total):
            line = file.readline().strip().split()
            if (mapping[n]>=0):
                mesh.elem2D = mesh.elem2D.at[0, mapping[n]].set(int(line[0]) - 1)
                mesh.elem2D = mesh.elem2D.at[1, mapping[n]].set(int(line[1]) - 1)
                mesh.elem2D = mesh.elem2D.at[2, mapping[n]].set(int(line[2]) - 1)

    # Convert global to local numbering
    mapping = mapping.at[:mesh.nod2D].set(-1)
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        ipos = partit.myList_nod2D[n]-1
        mapping = mapping.at[ipos].set(n)


    #print("elem 876 test:", partit.mype, jnp.any(partit.myList_elem2D == 876))
    #raise SystemExit("STOP HERE")

    for n in range(partit.myDim_elem2D):# + partit.eDim_elem2D):# + partit.eXDim_elem2D):
        for m in range(3):
            nn = mesh.elem2D[m, n]
            mesh.elem2D = mesh.elem2D.at[m, n].set(mapping[nn])

#    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
#        print(mesh.elem2D[:, n])
    if partit.mype == 0:
        print("elements are read")

    # Read 3D auxiliary data
    file_name = meshpath.strip() + '/aux3d.out'
    mesh.nl = jnp.zeros(1, dtype=jnp.int32)
    with open(file_name, 'r') as file:
        if partit.mype == 0:
            mesh.nl = int(file.readline().strip())  # Read the number of levels
        # Broadcast the number of levels to all processes
        mesh.nl = partit.MPI_COMM_FESOM.bcast(mesh.nl, root=0)

        # Check if the number of levels is less than 3
        if mesh.nl < 3:
            if partit.mype == 0:
                print("!!!Number of levels is less than 3, model will stop!!!")
            MPI.COMM_WORLD.Abort(1)  # Stop execution

        # Allocate the array for storing the standard depths
        mesh.zbar = jnp.zeros(mesh.nl, dtype=jnp.float32)

        # Read the standard depths on the root process
        if partit.mype == 0:
            file.readline()  # Skip the first line (already read)
            mesh.zbar = jnp.array([float(val) for val in file.readline().strip().split()])

        # Broadcast the zbar array to all processes
        mesh.zbar = partit.MPI_COMM_FESOM.bcast(mesh.zbar, root=0)

        # Ensure zbar is negative
        if mesh.zbar[1] > 0:
            mesh.zbar = -mesh.zbar

        # Allocate the array for mid-depths of cells
        mesh.Z = 0.5 * (mesh.zbar[:-1] + mesh.zbar[1:])

        # Allocate the array for depth information
        mesh.depth = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.float32)

        # Initialize mesh.check
        mesh.check = 0

        # Process the data in chunks
        for nchunk in range((mesh.nod2D - 1) // chunk_size + 1):
            mapping = jnp.zeros(chunk_size, dtype=jnp.int32)

        # Create the mapping for the current chunk
        for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
            ipos = (partit.myList_nod2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit.myList_nod2D[n] - nchunk * chunk_size - 1
                mapping = mapping.at[iofs].set(n + 1)  # Using 1-based indexing similar to Fortran

        # Determine the number of elements to read in this chunk
        k = min(chunk_size, mesh.nod2D - nchunk * chunk_size)

        # Read the depth values into the buffer on the root process
        if partit.mype == 0:
            rbuff = np.zeros((k, 1), dtype=np.float32)
            for n in range(k):
                rbuff[n, 0] = float(file.readline().strip())

            # Broadcast the buffer to all processes
        rbuff = partit.MPI_COMM_FESOM.bcast(rbuff, root=0)
        count = 0
        # Process the depths
        for n in range(k):
            x = rbuff[n, 0]
            if x > 0:
                x = -x  # Depths must be negative
            if x > -20.:  # Adjust based on threshold
                x = -20.
            if mapping[n] > 0:
                mesh.check += 1
                mesh.depth = mesh.depth.at[mapping[n] - 1].set(x)  # Adjust back to 0-based indexing
                count = count + 1

    # ==============================
    # Communication information
    # Every process reads its file
    # ==============================
    MAX_NEIGHBOR_PARTITIONS = 10
    max_neighbor_partitions = 10
    # Set file paths
    file_name = f"{dist_mesh_dir.strip()}/com_info{str(partit.mype).zfill(5)}.out"
    with open(file_name, 'r') as file:
        # Read the number of nodes
        n = int(file.readline().strip())

        # Read and validate rPEnum for nodes
        partit.com_nod2D.rPEnum = int(file.readline().strip())
        if partit.com_nod2D.rPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

        # Read rPE
        partit.com_nod2D.rPE = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_nod2D.rPEnum])

        # Read rptr
        partit.com_nod2D.rptr = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_nod2D.rPEnum + 1])

        # Allocate and read rlist
        partit.com_nod2D.rlist = jnp.zeros(partit.eDim_nod2D, dtype=jnp.int32)
        count = 0
        while count < partit.eDim_nod2D:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_nod2D.rlist = partit.com_nod2D.rlist.at[count:count + length].set(jnp.array(values))
            count += length

        # Read and validate sPEnum for nodes
        partit.com_nod2D.sPEnum = int(file.readline().strip())
        if partit.com_nod2D.sPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")
        # Read sPE
        partit.com_nod2D.sPE = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_nod2D.sPEnum])

        # Read sptr
        partit.com_nod2D.sptr = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_nod2D.sPEnum + 1])

        # Allocate and read slist
        n_slist = partit.com_nod2D.sptr[-1] - 1
        partit.com_nod2D.slist = jnp.zeros(n_slist, dtype=jnp.int32)
        count = 0
        while count < n_slist:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_nod2D.slist = partit.com_nod2D.slist.at[count:count + length].set(jnp.array(values))
            count += length
        # Read and validate rPEnum for elements
        partit.com_elem2D.rPEnum = int(file.readline().strip())
        if partit.com_elem2D.rPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

        # Read rPE
        partit.com_elem2D.rPE = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_elem2D.rPEnum])

        # Read rptr
        partit.com_elem2D.rptr = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D.rPEnum + 1])

        # Allocate and read rlist
        partit.com_elem2D.rlist = jnp.zeros(partit.eDim_elem2D, dtype=jnp.int32)
        count = 0
        while count < partit.eDim_elem2D:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_elem2D.rlist = partit.com_elem2D.rlist.at[count:count + length].set(jnp.array(values))
            count += length

        # Read and validate sPEnum for elements
        partit.com_elem2D.sPEnum = int(file.readline().strip())
        if partit.com_elem2D.sPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

        # Read sPE
        partit.com_elem2D.sPE = jnp.array(list(map(int, file.readline().strip().split()))[:partit.com_elem2D.sPEnum])

        # Read sptr
        partit.com_elem2D.sptr = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D.sPEnum + 1])

        # Allocate and read slist
        n_slist = partit.com_elem2D.sptr[-1] - 1
        partit.com_elem2D.slist = jnp.zeros(n_slist, dtype=jnp.int32)
        count = 0
        while count < n_slist:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_elem2D.slist = partit.com_elem2D.slist.at[count:count + length].set(jnp.array(values))
            count += length

        # Read and validate rPEnum for full elements
        partit.com_elem2D_full.rPEnum = int(file.readline().strip())
        if partit.com_elem2D_full.rPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

        # Read rPE
        partit.com_elem2D_full.rPE = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D_full.rPEnum])

        # Read rptr
        partit.com_elem2D_full.rptr = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D_full.rPEnum + 1])

        # Allocate and read rlist
        partit.com_elem2D_full.rlist = jnp.zeros(partit.eDim_elem2D + partit.eXDim_elem2D, dtype=jnp.int32)
        count = 0
        while count < partit.eDim_elem2D + partit.eXDim_elem2D:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_elem2D_full.rlist = partit.com_elem2D_full.rlist.at[count:count + length].set(jnp.array(values))
            count += length

        # Read and validate sPEnum for full elements
        partit.com_elem2D_full.sPEnum = int(file.readline().strip())
        if partit.com_elem2D_full.sPEnum > max_neighbor_partitions:
            raise ValueError("Increase MAX_NEIGHBOR_PARTITIONS in gen_modules_partitioning.F90 and recompile")

        # Read sPE
        partit.com_elem2D_full.sPE = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D_full.sPEnum])

        # Read sptr
        partit.com_elem2D_full.sptr = jnp.array(
            list(map(int, file.readline().strip().split()))[:partit.com_elem2D_full.sPEnum + 1])

        # Allocate and read slist
        n_slist = partit.com_elem2D_full.sptr[-1] - 1
        partit.com_elem2D_full.slist = jnp.zeros(n_slist, dtype=jnp.int32)
        count = 0
        while count < n_slist:
            values = list(map(int, file.readline().strip().split()))
            length = len(values)
            partit.com_elem2D_full.slist = partit.com_elem2D_full.slist.at[count:count + length].set(jnp.array(values))
            count += length

    if partit.mype == 0:
        print("Communication arrays are read")

    del rbuff
    del ibuff
    del mapping
    print(partit.mype, partit.eDim_nod2D, partit.com_nod2D.rlist)

    if partit.mype == 0:
        if partit.npes > 1:
            # Allocate remPtr arrays
            partit.remPtr_nod2D = jnp.zeros(partit.npes, dtype=jnp.int32)
            partit.remPtr_elem2D = jnp.zeros(partit.npes, dtype=jnp.int32)

            # Initialize remPtr arrays
            partit.remPtr_nod2D = partit.remPtr_nod2D.at[0].set(1)
            partit.remPtr_elem2D = partit.remPtr_elem2D.at[0].set(1)

            for n in range(1, partit.npes):
                # Receive n2D and e2D from other processes
                n2D = partit.MPI_COMM_FESOM.recv(source=n, tag=0)
                e2D = partit.MPI_COMM_FESOM.recv(source=n, tag=1)
                print("expected n2d, e2d=", n2D, e2D)
                # Update remPtr arrays
                partit.remPtr_nod2D = partit.remPtr_nod2D.at[n].set(partit.remPtr_nod2D[n - 1] + n2D)
                partit.remPtr_elem2D = partit.remPtr_elem2D.at[n].set(partit.remPtr_elem2D[n - 1] + e2D)

            # Allocate remList arrays
            partit.remList_nod2D = jnp.zeros(partit.remPtr_nod2D[-1], dtype=jnp.int32)
            partit.remList_elem2D = jnp.zeros(partit.remPtr_elem2D[-1], dtype=jnp.int32)

            for n in range(1, partit.npes):
                # Receive nod2D and elem2D lists from other processes
                nstart = partit.remPtr_nod2D[n - 1]
                n2D = partit.remPtr_nod2D[n] - nstart
                partit.remList_nod2D = partit.remList_nod2D.at[nstart:nstart + n2D].set(
                    partit.MPI_COMM_FESOM.recv(source=n, tag=2))

                estart = partit.remPtr_elem2D[n - 1]
                e2D = partit.remPtr_elem2D[n] - estart
                partit.remList_elem2D = partit.remList_elem2D.at[estart:estart + e2D].set(
                    partit.MPI_COMM_FESOM.recv(source=n, tag=3))

    else:
        # Send myDim_nod2D and myDim_elem2D to process 0
        partit.MPI_COMM_FESOM.send(partit.myDim_nod2D, dest=0, tag=0)
        partit.MPI_COMM_FESOM.send(partit.myDim_elem2D, dest=0, tag=1)

        # Send myList_nod2D and myList_elem2D to process 0
        partit.MPI_COMM_FESOM.send(partit.myList_nod2D[:partit.myDim_nod2D], dest=0, tag=2)
        partit.MPI_COMM_FESOM.send(partit.myList_elem2D[:partit.myDim_elem2D], dest=0, tag=3)

    return mesh, partit

# Assuming `trim_cyclic` is already defined as in the previous example
def trim_cyclic(b, cyclic_length):
    return jnp.where(b > cyclic_length / 2, b - cyclic_length,
                     jnp.where(b < -cyclic_length / 2, b + cyclic_length, b))


def test_tri(mesh, partit, cyclic_length):
    for n in range(partit.myDim_elem2D):
        elnodes = mesh.elem2D[:, n]

        # Extract and calculate vectors a, b, and c
        a = mesh.coord_nod2D[:, elnodes[0]]
        b = mesh.coord_nod2D[:, elnodes[1]] - a
        c = mesh.coord_nod2D[:, elnodes[2]] - a

        # Apply cyclic trimming to the first component of b and c
        b = b.at[0].set(trim_cyclic(b[0], cyclic_length))
        c = c.at[0].set(trim_cyclic(c[0], cyclic_length))

        # Compute r to check node order
        r = b[0] * c[1] - b[1] * c[0]

        if r > 0:
            # Swap second and third nodes if necessary
            nx = elnodes[1]
            elnodes = elnodes.at[1].set(elnodes[2])
            elnodes = elnodes.at[2].set(nx)

            # Update the mesh element node connectivity
            mesh.elem2D = mesh.elem2D.at[:, n].set(elnodes)
    return mesh, partit

    if partit.mype == 0:
        print('test_tri finished')
        print('=========================')


import jax.numpy as jnp
from mpi4py import MPI


def load_edges(mesh, partit,meshpath):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    chunk_size = 100000
    mesh_check = 0

    # Step 1: Edge array is already available, read edge2D and edge2D_in
    if mype == 0:
        print(f"reading {meshpath}/edgenum.out")
        with open(f"{meshpath}/edgenum.out", 'r') as file:
            mesh.edge2D = int(file.readline().strip())
            mesh.edge2D_in = int(file.readline().strip())
            print(f"2D mesh info : edge2D = {mesh.edge2D}")

    mesh.edge2D=comm.bcast(mesh.edge2D, root=0)
    mesh.edge2D_in=comm.bcast(mesh.edge2D_in, root=0)

    mesh.edges = jnp.zeros((2, partit.myDim_edge2D + partit.eDim_edge2D), dtype=jnp.int32)
    mesh.edge_tri = jnp.zeros((2, partit.myDim_edge2D + partit.eDim_edge2D), dtype=jnp.int32)
    # Step 2: Read edges and edge_tri from files in chunks and distribute them
    ibuff = jnp.zeros((4, chunk_size), dtype=jnp.int32)
    print(mype, mesh.edge2D, mesh.edge2D_in)
    if mype == 0:
        print(f"reading {meshpath}/edges.out")
        edges_file = open(f"{meshpath}/edges.out", 'r')
        edge_tri_file = open(f"{meshpath}/edge_tri.out", 'r')
    mapping = jnp.zeros(chunk_size, dtype=jnp.int32)
    for nchunk in range((mesh.edge2D - 1) // chunk_size + 1):
        mapping = mapping.at[:chunk_size].set(-1)
        for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
            ipos = (partit.myList_edge2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit.myList_edge2D[n] - nchunk * chunk_size -1 
                mapping = mapping.at[iofs].set(n)
        k = min(chunk_size, mesh.edge2D - nchunk * chunk_size)
        if mype == 0:
            for n in range(k):
                ibuff = ibuff.at[:, n].set(jnp.array([int(x) for x in edges_file.readline().split()] +
                                                     [int(x) for x in edge_tri_file.readline().split()]))
        ibuff=comm.bcast(ibuff[:, :k], root=0)
        for n in range(k):
            if mapping[n] >= 0:
                mesh_check += 1
                mesh.edges = mesh.edges.at[:, mapping[n]].set(ibuff[0:2, n])
                mesh.edge_tri = mesh.edge_tri.at[:, mapping[n]].set(ibuff[2:4, n])
    if mesh_check != partit.myDim_edge2D + partit.eDim_edge2D:
        print(f"ERROR while reading edges.out/edge_tri.out on mype = {mype}")
        print(
            f"{mesh_check} values have been read, but it does not equal myDim_edge2D + eDim_edge2D = {partit.myDim_edge2D + partit.eDim_edge2D}")

    if mype == 0:
        edges_file.close()
        edge_tri_file.close()

    # Step 3: Transform edge nodes to local indexing
    mesh_check = 0
    for nchunk in range((mesh.nod2D - 1) // chunk_size + 1):
        mapping = mapping.at[:chunk_size].set(0)

        for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
            ipos = (partit.myList_nod2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit.myList_nod2D[n] - nchunk * chunk_size - 1
                mapping = mapping.at[iofs].set(n)

        for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
            for m in range(2):
                nn = mesh.edges[m, n]
                ipos = (nn - 1) // chunk_size
                if ipos == nchunk:
                    mesh_check += 1
                    iofs = nn - nchunk * chunk_size -1 
                    mesh.edges = mesh.edges.at[m, n].set(-mapping[iofs]-1)

    mesh.edges = -mesh.edges-1
    mesh_check //= 2
    if mesh_check != partit.myDim_edge2D + partit.eDim_edge2D:
        print(f"ERROR while transforming edge nodes to local indexing on mype = {mype}")
        print(
            f"{mesh_check} edges have been transformed, but it does not equal myDim_edge2D + eDim_edge2D = {partit.myDim_edge2D + partit.eDim_edge2D}")

    # Step 4: Transform edge_tri to local indexing
    mesh_check = 0
    for nchunk in range(0, (mesh.elem2D.shape[0] + chunk_size - 1) // chunk_size):
        mapping = mapping.at[:chunk_size].set(0)

        for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
            ipos = (partit.myList_elem2D[n] - 1) // chunk_size
            if ipos == nchunk:
                iofs = partit.myList_elem2D[n] - nchunk * chunk_size - 1
                mapping = mapping.at[iofs].set(n)
        mesh.edge_tri = jnp.where(mesh.edge_tri < 0, 0, mesh.edge_tri)
        for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
            for m in range(2):
                nn = mesh.edge_tri[m, n]
                ipos = (nn - 1) // chunk_size
                if ipos == nchunk and nn > 0:
                    mesh_check += abs(m - 1)
                    iofs = nn - nchunk * chunk_size - 1
                    mesh.edge_tri = mesh.edge_tri.at[m, n].set(-mapping[iofs]-1)

    mesh.edge_tri = -mesh.edge_tri-1
    if mesh_check != partit.myDim_edge2D + partit.eDim_edge2D:
        print(f"ERROR while transforming edge elements to local indexing on mype = {mype}")
        print(
            f"{mesh_check} edges have been transformed, but it does not equal myDim_edge2D + eDim_edge2D = {partit.myDim_edge2D + partit.eDim_edge2D}")

    # Step 5: Build elem_edges from edge_tri
    mesh.elem_edges = jnp.zeros((3, partit.myDim_elem2D), dtype=jnp.int32)
    aux = jnp.zeros(partit.myDim_elem2D, dtype=jnp.int32)

    for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
        for k in range(2):
            q = mesh.edge_tri[k, n]
            if 0 < q <= partit.myDim_elem2D:
                aux = aux.at[q - 1].add(1)
                mesh.elem_edges = mesh.elem_edges.at[aux[q - 1] - 1, q - 1].set(n)

    # Step 6: Ensure edges are listed in the same rotation sense as nodes
    for elem in range(partit.myDim_elem2D):
        elnodes = mesh.elem2D[:, elem]
        eledges = mesh.elem_edges[:, elem]

        for q in range(3):
            for k in range(3):
                if (mesh.edges[0, eledges[k]] != elnodes[q]) and (mesh.edges[1, eledges[k]] != elnodes[q]):
                    mesh.elem_edges = mesh.elem_edges.at[q, elem].set(eledges[k])
                    break

    print(f"load_edges finished on mype = {mype}")
    return mesh, partit

def edge_center(n1, n2, mesh):
    """
    Calculate the center of an edge formed by nodes n1 and n2.
    Adjusts coordinates for cyclic length.
    """
    a = mesh.coord_nod2D[:, n1]
    b = mesh.coord_nod2D[:, n2]

    if a[0] - b[0] > cyclic_length / 2.0:
        a = a.at[0].set(a[0] - cyclic_length)
    if a[0] - b[0] < -cyclic_length / 2.0:
        b = b.at[0].set(b[0] - cyclic_length)

    x = 0.5 * (a[0] + b[0])
    y = 0.5 * (a[1] + b[1])

    return x, y

def elem_center(elem, mesh):
    """
    Calculate the center of an element.
    Adjust coordinates for cyclic length.
    """
    elnodes = mesh.elem2D[:, elem]
    ax = mesh.coord_nod2D[0, elnodes]
    amin = jnp.min(ax)

    for k in range(3):
        if ax[k] - amin >= cyclic_length / 2.0:
            ax = ax.at[k].set(ax[k] - cyclic_length)
        elif ax[k] - amin < -cyclic_length / 2.0:
            ax = ax.at[k].set(ax[k] + cyclic_length)

    x = jnp.sum(ax) / 3.0
    y = jnp.sum(mesh.coord_nod2D[1, elnodes]) / 3.0

    return x, y


def exchange_nod2D(nod_array2D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array2D to NumPy array if necessary (to ensure it's writable)
    nod_array2D_np = np.array(nod_array2D, dtype=np.float64, copy=True)

    # Buffers for send and receive operations
    s_buff_nod2D = [None] * sn
    r_buff_nod2D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer
    for n in range(sn):
        nini = com_nod2D.sptr[n]
        nend = com_nod2D.sptr[n + 1] - 1
        s_buff_nod2D[n] = nod_array2D_np[com_nod2D.slist[nini:nend + 1]-1]

    # Non-blocking MPI send
    for n in range(sn):
        dest = com_nod2D.sPE[n]
        nini = com_nod2D.sptr[n]
        offset = com_nod2D.sptr[n + 1] - nini
        req = comm.Isend(s_buff_nod2D[n], dest=dest, tag=mype)
        sreq.append(req)

    # Non-blocking MPI receive
    for n in range(rn):
        source = com_nod2D.rPE[n]
        nini = com_nod2D.rptr[n]
        offset = com_nod2D.rptr[n + 1] - nini
        r_buff_nod2D[n] = np.zeros(offset, dtype=np.float64)
        req = comm.Irecv(r_buff_nod2D[n], source=source, tag=source)
        rreq.append(req)

    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)

    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_nod2D.rptr[n]
        nend = com_nod2D.rptr[n + 1] - 1
#       print("size check:", mype, n, nini, nend, len(com_nod2D.rlist[nini:nend + 1]), len(r_buff_nod2D[n]))

        nod_array2D_np[com_nod2D.rlist[nini:nend + 1]-1] = r_buff_nod2D[n]

    # Optionally convert back to JAX array if necessary
    nod_array2D = jnp.array(nod_array2D_np)

    return nod_array2D

def exchange_nod2D_i(nod_array2D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array2D to NumPy array if necessary (to ensure it's writable)
    nod_array2D_np = np.array(nod_array2D, dtype=np.int32, copy=True)

    # Buffers for send and receive operations
    s_buff_nod2D = [None] * sn
    r_buff_nod2D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []
    # Prepare the send buffer
    for n in range(sn):
        nini = com_nod2D.sptr[n]-1
        nend = com_nod2D.sptr[n + 1] - 2
        s_buff_nod2D[n] = nod_array2D_np[com_nod2D.slist[nini:nend + 1]-1]

    # Non-blocking MPI send
    for n in range(sn):
        dest = com_nod2D.sPE[n]
        nini = com_nod2D.sptr[n]
        offset = com_nod2D.sptr[n + 1] - nini
        req = comm.Isend(s_buff_nod2D[n], dest=dest, tag=mype)
        sreq.append(req)

    # Non-blocking MPI receive
    for n in range(rn):
        source = com_nod2D.rPE[n]
        nini = com_nod2D.rptr[n]
        offset = com_nod2D.rptr[n + 1] - nini
        r_buff_nod2D[n] = np.zeros(offset, dtype=np.int32)
        req = comm.Irecv(r_buff_nod2D[n], source=source, tag=source)
        rreq.append(req)

    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)

    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_nod2D.rptr[n]-1
        nend = com_nod2D.rptr[n + 1] - 2
#       print("size check:", mype, n, nini, nend, len(com_nod2D.rlist[nini:nend + 1]), len(r_buff_nod2D[n]))
#        print("in exchange before:", mype, nod_array2D_np[com_nod2D.rlist[nini:nend + 1]-1])
        nod_array2D_np[com_nod2D.rlist[nini:nend + 1]-1] = r_buff_nod2D[n]
#        print("in exchange after:", mype, nod_array2D_np[com_nod2D.rlist[nini:nend + 1]-1])
#        print("in exchange rbuff:", mype, r_buff_nod2D[n])

    # Optionally convert back to JAX array if necessary
    nod_array2D = jnp.array(nod_array2D_np)

    return nod_array2D


def find_neighbors(mesh, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()

    # Allocate memory for element neighbors and initialize to 0
    mesh.elem_neighbors = jnp.zeros((3, partit.myDim_elem2D), dtype=jnp.int32)
    # Find element neighbors that share edges
    for elem in range(partit.myDim_elem2D):
        eledges = mesh.elem_edges[:, elem]
        for j in range(3):
            elem1 = mesh.edge_tri[0, eledges[j]]
            if elem1 == elem:
                elem1 = mesh.edge_tri[1, eledges[j]]
            mesh.elem_neighbors = mesh.elem_neighbors.at[j, elem].set(elem1)

    # Node neighborhood: find elements containing each node
    mesh.nod_in_elem2D_num = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)

    for n in range(partit.myDim_elem2D):
        for j in range(3):
            node = mesh.elem2D[j, n]
            if node >= partit.myDim_nod2D:
                continue
            mesh.nod_in_elem2D_num = mesh.nod_in_elem2D_num.at[node].add(1)

    # Synchronize processes
    comm.Barrier()
    # Find the max number of elements associated with each node across processes
    mymax = jnp.zeros(npes, dtype=jnp.int32)
    rmax = jnp.zeros(npes, dtype=jnp.int32)
    mymax = mymax.at[mype].set(jnp.max(mesh.nod_in_elem2D_num[:partit.myDim_nod2D]))

    mymax_np = np.asarray(mymax, dtype=np.int32)
    rmax_np = np.zeros_like(mymax_np, dtype=np.int32)
    comm.Allreduce(mymax_np, rmax_np, op=MPI.SUM)
    mymax = jnp.array(mymax_np)
    rmax = jnp.array(rmax_np)

    # Allocate nod_in_elem2D array and reset values
    max_rmax = jnp.max(rmax)
    mesh.nod_in_elem2D = jnp.zeros((max_rmax, partit.myDim_nod2D + partit.eDim_nod2D), dtype=jnp.int32)
    mesh.nod_in_elem2D_num = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    # Fill nod_in_elem2D array with the elements containing the node
    count=0
    for n in range(partit.myDim_elem2D):
        for j in range(3):
            node = mesh.elem2D[j, n]
            if node >= partit.myDim_nod2D:
                continue
            mesh.nod_in_elem2D_num = mesh.nod_in_elem2D_num.at[node].add(1)
            mesh.nod_in_elem2D = mesh.nod_in_elem2D.at[mesh.nod_in_elem2D_num[node] - 1, node].set(n)
    print("jnp.min/max(mesh.nod_in_elem2D_num) before =", partit.mype, jnp.min(mesh.nod_in_elem2D_num), jnp.max(mesh.nod_in_elem2D_num), partit.myDim_elem2D)
    # Exchange nod_in_elem2D_num between processors
    mesh.nod_in_elem2D_num=exchange_nod2D_i(mesh.nod_in_elem2D_num, partit)
    print("jnp.min/max(mesh.nod_in_elem2D_num) after =", partit.mype, jnp.min(mesh.nod_in_elem2D_num), jnp.max(mesh.nod_in_elem2D_num), partit.myDim_elem2D)
    
    # Temporary array for global element numbers
    temp_i = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    for n in range(max_rmax):
#       print(mype, "find_neighbors n/max_rmax=",n,max_rmax)
        for j in range(partit.myDim_nod2D):
            if mesh.nod_in_elem2D[n, j] >= 0:
                temp_i = temp_i.at[j].set(partit.myList_elem2D[mesh.nod_in_elem2D[n, j]])
        temp_i=exchange_nod2D_i(temp_i, partit)
        mesh.nod_in_elem2D = mesh.nod_in_elem2D.at[n, :].set(temp_i)

    # Substitute back local element numbers
    temp_i = jnp.zeros(mesh.elem2D.shape[1], dtype=jnp.int32)
    for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
        temp_i = temp_i.at[partit.myList_elem2D[n]].set(n)

    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        for j in range(mesh.nod_in_elem2D_num[n].item()):
            mesh.nod_in_elem2D = mesh.nod_in_elem2D.at[j, n].set(temp_i[mesh.nod_in_elem2D[j, n]])
    # Validate that each element has at least two valid neighbors
    for elem in range(partit.myDim_elem2D):
        elem1 = 0
        for j in range(3):
            if mesh.elem_neighbors[j, elem] >= 0:
                elem1 += 1

        if elem1 < 2:
            print(f"Insufficient number of neighbors for element {partit.myList_elem2D[elem]}")
            comm.Abort(1)
    print(mype, "find_neighbors part finished")
    return mesh, partit
