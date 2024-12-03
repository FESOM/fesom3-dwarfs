import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from mpi4jax import send, recv, bcast
from module_rotate_grid import *

def read_mesh_and_partition(mesh, partit, meshpath, force_rotation):
    dist_mesh_dir = meshpath + 'dist_' + str(partit.npes) + '/'
    file_name = dist_mesh_dir.strip() + '/rpart.out'

    # Initialize variables
    partit.npes = jnp.zeros(1, dtype=jnp.int32)  # Initialize as JAX array

    if partit.mype == 0:
        with open(file_name, 'r') as file:
            # Read the number of processors
            partit.npes = int(int(file.readline().strip()))
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
        partit.myDim_nod2D = int(file.readline().strip())

        # Read partit%eDim_nod2D
        partit.eDim_nod2D = int(file.readline().strip())

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
        partit.myDim_elem2D = int(file.readline().strip())

        # Read partit%eDim_elem2D
        partit.eDim_elem2D = int(file.readline().strip())

        # Read partit%eXDim_elem2D
        partit.eXDim_elem2D = int(file.readline().strip())

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
        partit.myDim_edge2D = int(file.readline().strip())

        # Read partit%eDim_edge2D
        partit.eDim_edge2D = int(file.readline().strip())

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
    mapping = jnp.full(mesh.nod2D, -1, dtype=jnp.int32)
    # Allocate mesh.coord_nod2D with JAX
    mesh.coord_nod2D = jnp.zeros((2, partit.myDim_nod2D + partit.eDim_nod2D))
    error_status = 0
    mesh.check = 0
    file_name = meshpath.strip() + '/nod2d.out'
    with open(file_name, 'r') as file:
        n = int(file.readline().strip())  # read nod2d
        if n != mesh.nod2D:
            error_status = 1  # set the error status for consistency between part and nod2d
            print('reading', file_name)
        if error_status != 0:
            print(n)
            print('error: mesh.nod2D != part[npes]', mesh.nod2D, n)
            MPI.COMM_WORLD.Abort(1)  # Stop execution if there's an error

        # Create the mapping for local indexing
        for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
            ipos = partit.myList_nod2D[n]-1
            mapping = mapping.at[ipos].set(n)

        for n in range(mesh.nod2D):
            line = file.readline().strip().split()
            i0, r0, r1, i1 = int(line[0]), float(line[1]), float(line[2]), int(line[3])
            # Apply the offset for longitude shift
            offset = 0.0
            if   r0 > 180.0:
                offset = -360.0
            elif r0 < -180.0:
                offset = 360.0
            x = r0 * np.pi / 180.
            y = r1 * np.pi / 180.

            if (force_rotation):
               x,y=g2r(x, y)

            if mapping[n] >= 0:
                mesh.check += 1
                mesh.coord_nod2D = mesh.coord_nod2D.at[0, mapping[n]].set(x)
                mesh.coord_nod2D = mesh.coord_nod2D.at[1, mapping[n]].set(y)

    # mesh.check_total = partit.MPI_COMM_FESOM.allreduce(mesh.check, op=MPI.SUM)
    print("nod2D reading check:", partit.mype, mesh.check - partit.myDim_nod2D - partit.eDim_nod2D)
    ##############################################################################
    del mapping
    
    # Read 2D element data
#   mesh.elem2D = jnp.zeros((3, partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D), dtype=jnp.int32)
    mesh.elem2D = jnp.full((3, partit.myDim_elem2D), -1, dtype=jnp.int32)
    file_name = meshpath.strip() + '/elem2d.out'
    with open(file_name, 'r') as file:
        mesh.elem2D_total = int(0)
        mesh.elem2D_total = int(file.readline().strip())  # Read the total number of elem2D
        mapping = jnp.zeros(mesh.elem2D_total, dtype=jnp.int32)
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
    del mapping
    if partit.mype == 0:
        print("elements are read")

    # Read 3D auxiliary data
    mapping = jnp.full(mesh.nod2D, -1, dtype=jnp.int32)
    file_name = meshpath.strip() + '/aux3d.out'
    mesh.nl = jnp.zeros(1, dtype=jnp.int32)
    with open(file_name, 'r') as file:
        mesh.nl = int(file.readline().strip())  # Read the number of levels
        # Check if the number of levels is less than 3
        if mesh.nl < 3:
            if partit.mype == 0:
                print("!!!Number of levels is less than 3, model will stop!!!")
            MPI.COMM_WORLD.Abort(1)  # Stop execution

        # Allocate the array for storing the standard depths
        mesh.zbar = jnp.zeros(mesh.nl)
    with open(file_name, 'r') as file:
        # Read the standard depths
        file.readline()  # Skip the first line (already read)
        mesh.zbar = jnp.array([float(val) for _ in range(mesh.nl) for val in file.readline().strip().split()])

        # Ensure zbar is negative
        if mesh.zbar[1] > 0:
            mesh.zbar = -mesh.zbar

        # Allocate the array for mid-depths of cells
        mesh.Z = 0.5 * (mesh.zbar[:-1] + mesh.zbar[1:])

        # Allocate the array for depth information
        mesh.depth = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)

        mesh.check = 0
        # Create the mapping for the current chunk
        for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
            ipos = (partit.myList_nod2D[n] - 1)
            mapping = mapping.at[ipos].set(n)  # Using 1-based indexing similar to Fortran

        # Read the depth values into the buffer
        count = 0
        for n in range(mesh.nod2D):
            z = float(file.readline().strip())
            # Process the depths
            if z > 0:
               z = -z     # Depths must be negative
            if z > -20.:  # Adjust based on threshold
               z = -20.
            if mapping[n] >= 0:
                mesh.check += 1
                mesh.depth = mesh.depth.at[mapping[n]].set(z)
    print("min/max depth=", partit.mype, mesh.zbar.min(), mesh.zbar.max(), mesh.depth.min(), mesh.depth.max())
    del mapping
    print("depth reading check:", partit.mype, mesh.check - partit.myDim_nod2D - partit.eDim_nod2D)
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


def load_edges(mesh, partit, meshpath):
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
    mapping = jnp.full(mesh.edge2D, -1, dtype=jnp.int32)

    mesh.edges = jnp.zeros((2, partit.myDim_edge2D + partit.eDim_edge2D), dtype=jnp.int32)
    mesh.edge_tri = jnp.zeros((2, partit.myDim_edge2D + partit.eDim_edge2D), dtype=jnp.int32)
    # Step 2: Read edges and edge_tri from files in chunks and distribute them
    print(mype, mesh.edge2D, mesh.edge2D_in)
    mesh_check=0
    print(f"reading {meshpath}/edges.out")
    edges_file = open(f"{meshpath}/edges.out", 'r')
    edge_tri_file = open(f"{meshpath}/edge_tri.out", 'r')
    for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
        ipos = partit.myList_edge2D[n] - 1
        mapping = mapping.at[ipos].set(n)
    for n in range(mesh.edge2D):
        ibuff = jnp.array([int(x) for x in edges_file.readline().split()] +
                                                     [int(x) for x in edge_tri_file.readline().split()])
        if mapping[n] >= 0:
            mesh_check += 1
            mesh.edges = mesh.edges.at[:, mapping[n]].set(ibuff[0:2]-1)
            mesh.edge_tri = mesh.edge_tri.at[:, mapping[n]].set(ibuff[2:4]-1)
    if mesh_check != partit.myDim_edge2D + partit.eDim_edge2D:
        print(f"ERROR while reading edges.out/edge_tri.out on mype = {mype}")
        print(
            f"{mesh_check} values have been read, but it does not equal myDim_edge2D + eDim_edge2D = {partit.myDim_edge2D + partit.eDim_edge2D}")

    edges_file.close()
    edge_tri_file.close()
    del mapping
    # Step 3: Transform edge nodes to local indexing
    mapping = jnp.zeros(mesh.nod2D, dtype=jnp.int32)
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        ipos = partit.myList_nod2D[n] - 1
        mapping = mapping.at[ipos].set(n)

    for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
        for m in range(2):
            nn = mesh.edges[m, n]
            mesh.edges = mesh.edges.at[m, n].set(mapping[nn])
    del mapping
    # Step 4: Transform edge_tri to local indexing
    mapping = jnp.zeros(mesh.elem2D_total, dtype=jnp.int32)
    mesh_check = 0
    for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
        ipos = partit.myList_elem2D[n] - 1
        mapping = mapping.at[ipos].set(n)

    mesh.edge_tri = jnp.where(mesh.edge_tri < 0, -1, mesh.edge_tri)
    for n in range(partit.myDim_edge2D + partit.eDim_edge2D):
        for m in range(2):
            nn = mesh.edge_tri[m, n]
            if nn >= 0:
                mesh_check += abs(m - 1)
                mesh.edge_tri = mesh.edge_tri.at[m, n].set(mapping[nn])

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

def edge_center(n1, n2, mesh, cyclic_length):
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

    return jnp.array([x, y])
@jax.jit
def elem_center(elem, elem2D, coord_nod2D, cyclic_length):
    """
    Calculate the center of an element.
    Adjust coordinates for cyclic length.
    """
    elnodes = elem2D[:, elem]
    ax = coord_nod2D[0, elnodes]
    amin = jnp.min(ax)

    # Adjust `ax` coordinates using JAX's array operations
    ax = jnp.where(ax - amin >= cyclic_length / 2.0, ax - cyclic_length, ax)
    ax = jnp.where(ax - amin < -cyclic_length / 2.0, ax + cyclic_length, ax)

    # Calculate the center coordinates
    x = jnp.sum(ax) / 3.0
    y = jnp.sum(coord_nod2D[1, elnodes]) / 3.0

    return jnp.array([x, y])


def exchange_nod2D(nod_array2D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array2D to NumPy array if necessary (to ensure it's writable)
    nod_array2D_np = np.array(nod_array2D, copy=True)

    # Buffers for send and receive operations
    s_buff_nod2D = [None] * sn
    r_buff_nod2D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer
    for n in range(sn):
        nini = com_nod2D.sptr[n] - 1
        nend = com_nod2D.sptr[n + 1] - 2
        s_buff_nod2D[n] = nod_array2D_np[com_nod2D.slist[nini:nend+1]-1]

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
        r_buff_nod2D[n] = np.zeros(offset)
        req = comm.Irecv(r_buff_nod2D[n], source=source, tag=source)
        rreq.append(req)

    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)

    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_nod2D.rptr[n] - 1
        nend = com_nod2D.rptr[n + 1] - 2
        nod_array2D_np[com_nod2D.rlist[nini:nend+1]-1] = r_buff_nod2D[n]
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


def exchange_elem2D(elem_array2D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_elem2D = partit.com_elem2D

    # Get the number of send/receive processes
    sn = com_elem2D.sPEnum
    rn = com_elem2D.rPEnum

    # Convert elem_array2D to NumPy array if necessary (to ensure it's writable)
    elem_array2D_np = np.array(elem_array2D, copy=True)

    # Buffers for send and receive operations
    s_buff_elem2D = [None] * sn
    r_buff_elem2D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []
        # Prepare the send buffer
    for n in range(sn):
        nini = com_elem2D.sptr[n]-1
        nend = com_elem2D.sptr[n + 1] - 2
        s_buff_elem2D[n] = elem_array2D_np[com_elem2D.slist[nini:nend + 1]-1]

    # Non-blocking MPI send
    for n in range(sn):
        dest = com_elem2D.sPE[n]
        nini = com_elem2D.sptr[n]
        offset = com_elem2D.sptr[n + 1] - nini
        req = comm.Isend(s_buff_elem2D[n], dest=dest, tag=mype)
        sreq.append(req)

    # Non-blocking MPI receive
    for n in range(rn):
        source = com_elem2D.rPE[n]
        nini = com_elem2D.rptr[n]
        offset = com_elem2D.rptr[n + 1] - nini
        r_buff_elem2D[n] = np.zeros(offset)
        req = comm.Irecv(r_buff_elem2D[n], source=source, tag=source)
        rreq.append(req)
    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)
    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_elem2D.rptr[n]-1
        nend = com_elem2D.rptr[n + 1] - 2
        elem_array2D_np[com_elem2D.rlist[nini:nend + 1]-1] = r_buff_elem2D[n]

    # Optionally convert back to JAX array if necessary
    elem_array2D = jnp.array(elem_array2D_np)
    return elem_array2D

def exchange_nod3D(nod_array3D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array3D to NumPy array if necessary (to ensure it's writable)
    nod_array3D_np = np.array(nod_array3D, copy=True)
    nl1 = nod_array3D_np.shape[0]  # Size in the vertical dimension

    # Buffers for send and receive operations
    s_buff_nod3D = [None] * sn
    r_buff_nod3D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer
    for n in range(sn):
        nini = com_nod2D.sptr[n] - 1
        nend = com_nod2D.sptr[n + 1] - 2
        nc = 0
        s_buff_nod3D[n] = np.zeros((nl1 * (nend - nini + 1)))
        for nh in range(nini, nend + 1):
            for nz in range(nl1):
                s_buff_nod3D[n][nc] = nod_array3D_np[nz, com_nod2D.slist[nh] - 1]
                nc += 1

    # Non-blocking MPI send
    for n in range(sn):
        dest = com_nod2D.sPE[n]
        nini = com_nod2D.sptr[n]
        offset = (com_nod2D.sptr[n + 1] - nini) * nl1
        req = comm.Isend(s_buff_nod3D[n], dest=dest, tag=mype)
        sreq.append(req)

    # Non-blocking MPI receive
    for n in range(rn):
        source = com_nod2D.rPE[n]
        nini = com_nod2D.rptr[n]
        offset = (com_nod2D.rptr[n + 1] - nini) * nl1
        r_buff_nod3D[n] = np.zeros(offset)
        req = comm.Irecv(r_buff_nod3D[n], source=source, tag=source)
        rreq.append(req)

    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)

    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_nod2D.rptr[n] - 1
        nend = com_nod2D.rptr[n + 1] - 2
        nc = 0
        for nh in range(nini, nend + 1):
            for nz in range(nl1):
                nod_array3D_np[nz, com_nod2D.rlist[nh] - 1] = r_buff_nod3D[n][nc]
                nc += 1

    # Optionally convert back to JAX array if necessary
    nod_array3D = jnp.array(nod_array3D_np)

    return nod_array3D

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
    mesh.nod_in_elem2D = jnp.full((max_rmax, partit.myDim_nod2D + partit.eDim_nod2D), -1, dtype=jnp.int32)
    mesh.nod_in_elem2D_num = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    # Fill nod_in_elem2D array with the elements containing the node
    count=0
    for n in range(partit.myDim_elem2D):
        for j in range(3):
            node = mesh.elem2D[j, n]
            if node >= partit.myDim_nod2D:
                continue
            mesh.nod_in_elem2D_num = mesh.nod_in_elem2D_num.at[node].add(1)
            mesh.nod_in_elem2D     = mesh.nod_in_elem2D.at[mesh.nod_in_elem2D_num[node]-1, node].set(n)

    # Exchange nod_in_elem2D_num between processors
    mesh.nod_in_elem2D_num=exchange_nod2D_i(mesh.nod_in_elem2D_num, partit)
    # Temporary array for global element numbers
    temp_i = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    for n in range(max_rmax):
#       print(mype, "find_neighbors n/max_rmax=",n,max_rmax)
        for j in range(partit.myDim_nod2D):
            if mesh.nod_in_elem2D[n, j] >= 0:
                temp_i = temp_i.at[j].set(partit.myList_elem2D[mesh.nod_in_elem2D[n, j]]-1)
        temp_i=exchange_nod2D_i(temp_i, partit)
        mesh.nod_in_elem2D = mesh.nod_in_elem2D.at[n, :].set(temp_i)
    del temp_i
    if (mype == 0):
       print("super check 0:", mype, mesh.nod_in_elem2D[:, partit.myDim_nod2D-1])
    # Substitute back local element numbers
    temp_i = jnp.zeros(mesh.elem2D_total, dtype=jnp.int32)
    for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
        temp_i = temp_i.at[partit.myList_elem2D[n]-1].set(n)

    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        for j in range(mesh.nod_in_elem2D_num[n].item()):
            mesh.nod_in_elem2D = mesh.nod_in_elem2D.at[j, n].set(temp_i[mesh.nod_in_elem2D[j, n]])

    del temp_i
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

def find_levels(mesh, partit, meshpath):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()
    # Allocate nlevels and nlevels_nod2D
    mesh.nlevels = jnp.zeros(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D, dtype=jnp.int32)
    mesh.nlevels_nod2D = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    mapping = jnp.full(mesh.elem2D_total, -1, dtype=jnp.int32)

    for n in range(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D):
        ipos = partit.myList_elem2D[n] - 1
        mapping = mapping.at[ipos].set(n)

    elvls_file = open(f"{meshpath}/elvls.out", 'r')
    # Part I: Reading levels at elements
    for n in range(mesh.elem2D_total):
        elvls = int(elvls_file.readline().strip())-1
        if mapping[n] >= 0:
           mesh.nlevels = mesh.nlevels.at[mapping[n]].set(elvls)
    elvls_file.close()
    del mapping
    # Part II: Reading levels at nodes
    mapping = jnp.full(mesh.nod2D, -1, dtype=jnp.int32)
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        ipos = partit.myList_nod2D[n] - 1
        mapping = mapping.at[ipos].set(n)

    nlvls_file = open(f"{meshpath}/nlvls.out", 'r')
    for n in range(mesh.nod2D):
        nlvls = int(nlvls_file.readline().strip())-1
        if mapping[n] >= 0:
           mesh.nlevels_nod2D = mesh.nlevels_nod2D.at[mapping[n]].set(nlvls)

    # Allocate ulevels and ulevels_nod2D
    mesh.ulevels = jnp.ones(partit.myDim_elem2D + partit.eDim_elem2D + partit.eXDim_elem2D, dtype=jnp.int32)
    mesh.ulevels_nod2D = jnp.ones(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)

    # Print summary on mype = 0
    min_level = jnp.min(mesh.nlevels)
    max_level = jnp.max(mesh.nlevels)
    print(f"Min/max depth on mype {mype}: {mesh.zbar[min_level]}, {mesh.zbar[max_level]}")
#   print("mesh.zbar on mype", mype, mesh.zbar.min(), mesh.zbar.max())
    return mesh, partit

def find_levels_min_e2n(mesh, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()

    mesh.nlevels_nod2D_min = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    mesh.ulevels_nod2D_max = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)

    # Loop through all nodes and compute min/max levels for each node's neighboring elements
    for node in range(partit.myDim_nod2D):
        k = mesh.nod_in_elem2D_num[node]
        # Get the minimum depth in neighboring elements around node
        mesh.nlevels_nod2D_min = mesh.nlevels_nod2D_min.at[node].set(
            jnp.min(mesh.nlevels[mesh.nod_in_elem2D[:k, node]])
        )
        # Get the maximum u-levels in neighboring elements around node
        mesh.ulevels_nod2D_max = mesh.ulevels_nod2D_max.at[node].set(
            jnp.max(mesh.ulevels[mesh.nod_in_elem2D[:k, node]])
        )

    mesh.nlevels_nod2D_min = exchange_nod2D_i(mesh.nlevels_nod2D_min, partit)
    mesh.ulevels_nod2D_max = exchange_nod2D_i(mesh.ulevels_nod2D_max, partit)
    
    return mesh, partit


def mesh_areas(mesh, partit, cartesian, cyclic_length, r_earth):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()

    mesh.elem_area = jnp.zeros(partit.myDim_elem2D + partit.eDim_elem2D)
    mesh.area = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.areasvol = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.area_inv = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.areasvol_inv = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.mesh_resolution = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
    # Compute triangle areas
    for n in range(partit.myDim_elem2D):
        elnodes = mesh.elem2D[:, n]
        ay = jnp.sum(mesh.coord_nod2D[1, elnodes]) / 3.0
        ay = jnp.cos(ay) if not cartesian else 1.0
        a = mesh.coord_nod2D[:, elnodes[1]] - mesh.coord_nod2D[:, elnodes[0]]
        b = mesh.coord_nod2D[:, elnodes[2]] - mesh.coord_nod2D[:, elnodes[0]]
        a = trim_cyclic(a, cyclic_length)
        b = trim_cyclic(b, cyclic_length)
        a = a.at[0].set(a[0] * ay)
        b = b.at[0].set(b[0] * ay)
        mesh.elem_area = mesh.elem_area.at[n].set(0.5 * abs(a[0] * b[1] - b[0] * a[1]))
    # Exchange element areas
    mesh.elem_area = exchange_elem2D(mesh.elem_area, partit)

    elnodes = mesh.elem2D[:, 0]
    # Compute areas of upper/lower scalar cell edges
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        for j in range(mesh.nod_in_elem2D_num[n]):
            elem = mesh.nod_in_elem2D[j, n]
            nzmin = mesh.ulevels[elem]
            nzmax = mesh.nlevels[elem] - 1
            for nz in range(nzmin, nzmax + 1):
                mesh.area = mesh.area.at[nz, n].set(mesh.area[nz, n] + mesh.elem_area[elem] / 3.0)

    mesh.areasvol = mesh.area
    # Scale areas to meters squared
    mesh.elem_area *= r_earth * r_earth
    mesh.area *= r_earth * r_earth
    mesh.areasvol *= r_earth * r_earth
    # Exchange nodal areas
#    mesh.area     = exchange_nod3D(mesh.area,     partit)
#    mesh.areasvol = exchange_nod3D(mesh.areasvol, partit)
    if (mype==0):
        n=partit.myDim_nod2D-1
        for j in range(mesh.nod_in_elem2D_num[n]):
            elem = mesh.nod_in_elem2D[j, n]
        elem = mesh.nod_in_elem2D[j, n]
#    print("elem area check:", mype, mesh.elem_area[0], mesh.elem_area[partit.myDim_elem2D-1], jnp.sum(mesh.elem_area[:partit.myDim_elem2D]))
#    print("node area check:", mype, mesh.area[1,0], mesh.area[1,partit.myDim_nod2D-1], jnp.sum(mesh.area[1,:partit.myDim_nod2D]))

    # Compute inverse area
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        nzmin = mesh.ulevels_nod2D[n]
        nzmax = mesh.nlevels_nod2D[n]
        for nz in range(nzmin, nzmax + 1):
            mesh.area_inv = mesh.area_inv.at[nz, n].set(1.0 / mesh.area[nz, n] if mesh.area[nz, n] > 0.0 else 0.0)


    mesh.areasvol_inv = mesh.area_inv

    # Compute scalar cell resolution
    work_array = jnp.zeros(partit.myDim_nod2D)
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        mesh.mesh_resolution = mesh.mesh_resolution.at[n].set(jnp.sqrt(mesh.areasvol[mesh.ulevels_nod2D[n], n] / jnp.pi) * 2.0)

    # Smooth the resolution field
    for _ in range(3):  # Apply smoothing 3 times
        for n in range(partit.myDim_nod2D):
            vol = 0.0
            work_array = work_array.at[n].set(0.0)
            for j in range(mesh.nod_in_elem2D_num[n]):
                elem = mesh.nod_in_elem2D[j, n]
                elnodes = mesh.elem2D[:, elem]
                work_array = work_array.at[n].set(work_array[n] + jnp.sum(mesh.mesh_resolution[elnodes]) / 3.0 * mesh.elem_area[elem])
                vol += mesh.elem_area[elem]
            work_array = work_array.at[n].set(work_array[n] / vol)
        mesh_resolution = mesh.mesh_resolution.at[:partit.myDim_nod2D].set(work_array)
        mesh.mesh_resolution = exchange_nod2D(mesh.mesh_resolution, partit)
        comm.Barrier()
    # Compute total ocean areas with/without cavity
    vol = 0.0
    vol2 = 0.0
    print("1st level:", mesh.ulevels_nod2D.min(), mesh.ulevels_nod2D.max())
    for n in range(partit.myDim_nod2D):
        vol2 += mesh.areasvol[mesh.ulevels_nod2D[n], n]
        if mesh.ulevels_nod2D[n] == 1:
            vol += mesh.areasvol[1, n]

    mesh.ocean_area = comm.allreduce(vol, op=MPI.SUM)
    mesh.ocean_areawithcav = comm.allreduce(vol2, op=MPI.SUM)

    # Print mesh statistics on mype 0
    if mype == 0:
        print('____________________________________________________________________')
        print(f' --> mesh statistics (mype {mype}):')
        print(f'  MaxElemArea: {jnp.max(mesh.elem_area)}, MinElemArea: {jnp.min(mesh.elem_area)}')
        print(f'  MaxScalarArea: {jnp.max(mesh.area[0, :])}, MinScalarArea: {jnp.min(mesh.area[0, :])}')
        print(f'  Edges: {mesh.edge2D}, internal: {mesh.edge2D_in}')
        print(f'  Total ocean surface area: {mesh.ocean_area} m^2')
        print(f'  Total ocean surface area with cavity: {mesh.ocean_areawithcav} m^2')
    return mesh, partit

def mesh_auxiliary_arrays(mesh, partit, cartesian, fplane, cyclic_length, r_earth):
    """
    This function initializes auxiliary arrays for the mesh to accelerate gradient
    and divergence calculations, as well as to facilitate the handling of cyclicity.
    """
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()

    myDim_edge2D = partit.myDim_edge2D
    eDim_edge2D = partit.eDim_edge2D
    myDim_elem2D = partit.myDim_elem2D
    eDim_elem2D = partit.eDim_elem2D
    eXDim_elem2D = partit.eXDim_elem2D
    myDim_nod2D = partit.myDim_nod2D
    eDim_nod2D = partit.eDim_nod2D
    omega = 7.2921e-5  # Earth's rotation rate

    # Allocate arrays with JAX
    mesh.edge_dxdy = jnp.zeros((2, myDim_edge2D + eDim_edge2D))
    mesh.edge_cross_dxdy = jnp.zeros((4, myDim_edge2D + eDim_edge2D))
    mesh.gradient_sca = jnp.zeros((6, myDim_elem2D))
    mesh.gradient_vec = jnp.zeros((6, myDim_elem2D))
    mesh.metric_factor = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)
    mesh.elem_cos = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)
    mesh.coriolis = jnp.zeros(myDim_elem2D)
    mesh.coriolis_node = jnp.zeros(myDim_nod2D + eDim_nod2D)
    mesh.geo_coord_nod2D = jnp.zeros((2, myDim_nod2D + eDim_nod2D))
    center_x = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)
    center_y = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)

    # Compute Coriolis force at each node
    for n in range(myDim_nod2D + eDim_nod2D):
        lon, lat = r2g(mesh.coord_nod2D[0, n], mesh.coord_nod2D[1, n])
        mesh.coriolis_node = mesh.coriolis_node.at[n].set(2 * omega * jnp.sin(lat))
        if lon > 2 * jnp.pi:
            lon -= 2 * jnp.pi
        elif lon < -2 * jnp.pi:
            lon += 2 * jnp.pi
        mesh.geo_coord_nod2D = mesh.geo_coord_nod2D.at[:, n].set(jnp.array([lon, lat]))

    for n in range(myDim_elem2D):
        jnpaux = elem_center(n, mesh.elem2D, mesh.coord_nod2D, cyclic_length)
        ax=jnpaux[0]
        ay=jnpaux[1]
        lon, lat = r2g(ax, ay)
        mesh.coriolis = mesh.coriolis.at[n].set(2 * omega * jnp.sin(lat))
        center_x = center_x.at[n].set(ax)
        center_y = center_y.at[n].set(ay)
        mesh.elem_cos = mesh.elem_cos.at[n].set(jnp.cos(ay))
        mesh.metric_factor = mesh.metric_factor.at[n].set(jnp.tan(ay) / r_earth)

    if fplane:
        mesh.coriolis = mesh.coriolis.at[:].set(2 * omega * 0.71)

    # Exchange values across partitions
    mesh.metric_factor=exchange_elem2D(mesh.metric_factor, partit)
    mesh.elem_cos=exchange_elem2D(mesh.elem_cos, partit)
    center_x=exchange_elem2D(center_x, partit)
    center_y=exchange_elem2D(center_y, partit)

    if cartesian:
        mesh.elem_cos = 1.0
        mesh.metric_factor = 0.0

    # Compute distances along edges
    for n in range(myDim_edge2D + eDim_edge2D):
        ed = mesh.edges[:, n]
        a = mesh.coord_nod2D[:, ed[1]] - mesh.coord_nod2D[:, ed[0]]
        a = a.at[0].set(trim_cyclic(a[0], cyclic_length))
        mesh.edge_dxdy = mesh.edge_dxdy.at[:, n].set(a)

    # Compute cross distances for edges
    for n in range(myDim_edge2D + eDim_edge2D):
        ed = mesh.edges[:, n]
        el = mesh.edge_tri[:, n]
        a = edge_center(ed[0], ed[1], mesh, cyclic_length)
        b = jnp.array([center_x[el[0]], center_y[el[0]]]) - a
        b = b.at[0].set(trim_cyclic(b[0], cyclic_length))
        b = b.at[0].set(b[0]*mesh.elem_cos[el[0]])
        mesh.edge_cross_dxdy = mesh.edge_cross_dxdy.at[0:2, n].set(b * r_earth)
        if ((n==0) & (mype==0)):
            print("edgecheck", a, mesh.edge_cross_dxdy[0:2, n])
        if el[1] > 0:
            b = jnp.array([center_x[el[1]], center_y[el[1]]]) - jnp.array(a)
            b = b.at[0].set(trim_cyclic(b[0], cyclic_length))
            b = b.at[0].set(b[0] * mesh.elem_cos[el[1]])
            mesh.edge_cross_dxdy = mesh.edge_cross_dxdy.at[2:4, n].set(b * r_earth)
        else:
            mesh.edge_cross_dxdy = mesh.edge_cross_dxdy.at[2:4, n].set(0.0)

    # Compute derivatives of scalar quantities
    for elem in range(myDim_elem2D):
        elnodes = mesh.elem2D[:, elem]
        deltaX31 = mesh.coord_nod2D[0, elnodes[2]] - mesh.coord_nod2D[0, elnodes[0]]
        deltaX31=trim_cyclic(deltaX31,cyclic_length)
        deltaX31 *= mesh.elem_cos[elem]
        deltaX21 = mesh.coord_nod2D[0, elnodes[1]] - mesh.coord_nod2D[0, elnodes[0]]
        deltaX21=trim_cyclic(deltaX21,cyclic_length)
        deltaX21 *= mesh.elem_cos[elem]
        deltaY31 = mesh.coord_nod2D[1, elnodes[2]] - mesh.coord_nod2D[1, elnodes[0]]
        deltaY21 = mesh.coord_nod2D[1, elnodes[1]] - mesh.coord_nod2D[1, elnodes[0]]
        dfactor = -0.5 * r_earth / mesh.elem_area[elem]
        mesh.gradient_sca = mesh.gradient_sca.at[:, elem].set(jnp.array([
            (-deltaY31 + deltaY21) * dfactor,
            deltaY31 * dfactor,
            -deltaY21 * dfactor,
            (deltaX31 - deltaX21) * dfactor,
            -deltaX31 * dfactor,
            deltaX21 * dfactor
        ]))

    # Compute derivatives of vector quantities using least squares
    for elem in range(myDim_elem2D):
        a = jnp.array([center_x[elem], center_y[elem]])
        x, y = jnp.zeros(3), jnp.zeros(3)
        for j in range(3):
            el = mesh.elem_neighbors[j, elem]
            if el > 0:
                b = jnp.array([center_x[el], center_y[el]])
                x = x.at[j].set(b[0] - a[0])
                x = x.at[j].set(trim_cyclic(x[j], cyclic_length))
                y = y.at[j].set(b[1] - a[1])
            else:
                ed = mesh.edges[:, mesh.elem_edges[j, elem]]
                b = edge_center(ed[0], ed[1], mesh, cyclic_length)
                x = x.at[j].set(2 * (b[0] - a[0]))
                x = x.at[j].set(trim_cyclic(x[j], cyclic_length))
                y = y.at[j].set(2 * (b[1] - a[1]))
        x *= mesh.elem_cos[elem] * r_earth
        y *= r_earth
        cxx = jnp.sum(x ** 2)
        cxy = jnp.sum(x * y)
        cyy = jnp.sum(y ** 2)
        d = cxy ** 2 - cxx * cyy
        mesh.gradient_vec = mesh.gradient_vec.at[0:3, elem].set((cxy * y - cyy * x) / d)
        mesh.gradient_vec = mesh.gradient_vec.at[3:6, elem].set((cxy * x - cxx * y) / d)

    sum_X = jnp.sum(jnp.abs(jnp.sum(mesh.gradient_sca[0:3, :myDim_elem2D], 0)))
    sum_Y = jnp.sum(jnp.abs(jnp.sum(mesh.gradient_sca[3:6, :myDim_elem2D], 0)))
    sum_X_ABS = jnp.sum(jnp.sum(jnp.abs(mesh.gradient_sca[0:3, :myDim_elem2D]), 0))
    sum_Y_ABS = jnp.sum(jnp.sum(jnp.abs(mesh.gradient_sca[3:6, :myDim_elem2D]), 0))

    #print("gradient check x", mype, sum_X, sum_X_ABS)
    #print("gradient check y", mype, sum_Y, sum_Y_ABS)

    print("gradient supercheck1", mype, mesh.gradient_sca[:, 0])
    print("gradient supercheck2", mype, mesh.gradient_sca[:, myDim_elem2D-1])
    # Deallocate resources
    del center_x, center_y
    return mesh, partit


def init_ale(mesh, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    # Synchronize processes
    comm.Barrier()

    myDim_edge2D = partit.myDim_edge2D
    eDim_edge2D = partit.eDim_edge2D
    myDim_elem2D = partit.myDim_elem2D
    eDim_elem2D = partit.eDim_elem2D
    eXDim_elem2D = partit.eXDim_elem2D
    myDim_nod2D = partit.myDim_nod2D
    eDim_nod2D = partit.eDim_nod2D
    # Allocation of arrays
    mesh.hnode = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.hnode_new = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
    mesh.hbar = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
    mesh.hbar_old = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
    mesh.helem = jnp.zeros((mesh.nl - 1, partit.myDim_elem2D + partit.eDim_nod2D))
    mesh.dhe = jnp.zeros(partit.myDim_elem2D)
    mesh.zbar_3d_n = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))

    # Conditional allocation for asynchronous mode
    if partit.pe_status == 0:  # Assuming ib_async_mode is equivalent to pe_status
        mesh.Z_3d_n = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
        mesh.Z_3d_n_ib = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
    else:
        mesh.Z_3d_n = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
        mesh.Z_3d_n_ib = jnp.zeros((mesh.nl - 1, partit.myDim_nod2D + partit.eDim_nod2D))
        for i in range(partit.myDim_nod2D + partit.eDim_nod2D):
            for j in range(mesh.nl - 1):
                mesh.Z_3d_n[j, i] = 0.0
                mesh.Z_3d_n_ib[j, i] = 0.0

    mesh.bottom_elem_thickness = jnp.zeros(partit.myDim_elem2D + partit.eDim_nod2D)
    mesh.zbar_e_bot = jnp.zeros(partit.myDim_elem2D + partit.eDim_elem2D)
    mesh.zbar_e_srf = jnp.zeros(partit.myDim_elem2D + partit.eDim_elem2D)
    mesh.bottom_node_thickness = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
    mesh.zbar_n_bot = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
    mesh.zbar_n_srf = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)

    # Initialization of arrays
    mesh.hbar = jnp.zeros_like(mesh.hbar)
    mesh.hbar_old = jnp.zeros_like(mesh.hbar_old)
    mesh.dhe = jnp.zeros_like(mesh.dhe)
    mesh.hnode = jnp.zeros_like(mesh.hnode)
    mesh.hnode_new = jnp.zeros_like(mesh.hnode_new)
    mesh.helem = jnp.zeros_like(mesh.helem)

#    mesh.zbar_n_bot = jnp.zeros_like(mesh.zbar_n_bot)
#    mesh.zbar_e_bot = jnp.zeros_like(mesh.zbar_e_bot)
    mesh.zbar_n_srf = jnp.full_like(mesh.zbar_n_srf, mesh.zbar[0])
    mesh.zbar_e_srf = jnp.full_like(mesh.zbar_e_srf, mesh.zbar[0])

    for elem in range(partit.myDim_elem2D):
        nle = mesh.nlevels[elem]
        mesh.bottom_elem_thickness = mesh.bottom_elem_thickness.at[elem].set(mesh.zbar[nle - 1] - mesh.zbar[nle])
        mesh.zbar_e_bot = mesh.zbar_e_bot.at[elem].set(mesh.zbar[nle])

    for node in range(partit.myDim_nod2D):
        nln = mesh.nlevels_nod2D[node]
        mesh.zbar_n_bot = mesh.zbar_n_bot.at[node].set(mesh.zbar[nln])
        mesh.bottom_node_thickness = mesh.bottom_node_thickness.at[node].set(mesh.zbar[nln - 1] - mesh.zbar_n_bot[node])

    mesh.bottom_elem_thickness=exchange_elem2D(mesh.bottom_elem_thickness, partit)

    mesh.zbar_e_bot=exchange_elem2D(mesh.zbar_e_bot, partit)

    mesh.zbar_n_bot=exchange_nod2D(mesh.zbar_n_bot, partit)
    mesh.bottom_node_thickness=exchange_nod2D(mesh.bottom_node_thickness, partit)

    mesh.zbar_3d_n = jnp.zeros_like(mesh.zbar_3d_n)
    mesh.Z_3d_n = jnp.zeros_like(mesh.Z_3d_n)
    for n in range(partit.myDim_nod2D + partit.eDim_nod2D):
        nzmin = mesh.ulevels_nod2D[n]
        nzmax = mesh.nlevels_nod2D[n]

        # Updating zbar_3d_n and Z_3d_n arrays carefully
        mesh.zbar_3d_n = mesh.zbar_3d_n.at[0:nzmin, n].set(mesh.zbar[0:nzmin])
        mesh.zbar_3d_n = mesh.zbar_3d_n.at[nzmin-1, n].set(mesh.zbar_n_srf[n])
        mesh.zbar_3d_n = mesh.zbar_3d_n.at[nzmin:nzmax, n].set(mesh.zbar[nzmin:nzmax])
        mesh.zbar_3d_n = mesh.zbar_3d_n.at[nzmax-1, n].set(mesh.zbar_n_bot[n])

        mesh.Z_3d_n = mesh.Z_3d_n.at[0:nzmin, n].set(mesh.Z[0:nzmin])
        mesh.Z_3d_n = mesh.Z_3d_n.at[nzmin-1, n].set(
            mesh.zbar_3d_n[nzmin-1, n] + (mesh.zbar_3d_n[nzmin, n] - mesh.zbar_n_srf[n]) / 2
        )
        mesh.Z_3d_n = mesh.Z_3d_n.at[nzmin:nzmax - 2, n].set(mesh.Z[nzmin:nzmax - 2])
        mesh.Z_3d_n = mesh.Z_3d_n.at[nzmax - 2, n].set(
            mesh.zbar_3d_n[nzmax - 2, n] + (mesh.zbar_n_bot[n] - mesh.zbar_3d_n[nzmax - 2, n]) / 2
        )
    return mesh


def init_thickness_ale(mesh, partit):

    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes

    myDim_nod2D, eDim_nod2D   = partit.myDim_nod2D, partit.eDim_nod2D
    myDim_elem2D, eDim_elem2D = partit.myDim_elem2D, partit.eDim_elem2D
#    dynamics.ssh_rhs_old = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
#    dynamics.eta_n       = jnp.zeros_like(ssh_rhs_old)
#    print("mesh.zbar_3d_n=", mesh.zbar_3d_n)
# Linear Free-Surface
    for n in range(myDim_nod2D + eDim_nod2D):
        nzmin = mesh.ulevels_nod2D[n]
        nzmax = mesh.nlevels_nod2D[n] - 1

        # Set layer thicknesses
        for nz in range(nzmin-1, nzmax-1):
            mesh.hnode = mesh.hnode.at[nz, n].set(mesh.zbar_3d_n[nz, n] - mesh.zbar_3d_n[nz + 1, n])

        # Set bottom node thickness
        mesh.hnode = mesh.hnode.at[nzmax-1, n].set(mesh.bottom_node_thickness[n])

    for elem in range(myDim_elem2D):
        nzmin = mesh.ulevels[elem]
        nzmax = mesh.nlevels[elem] - 1

        # Set layer thicknesses
        mesh.helem = mesh.helem.at[nzmin-1, elem].set(mesh.zbar_e_srf[elem] - mesh.zbar[nzmin])
        for nz in range(nzmin, nzmax-1):
            mesh.helem = mesh.helem.at[nz, elem].set(mesh.zbar[nz] - mesh.zbar[nz + 1])

        # Set bottom element thickness
        mesh.helem = mesh.helem.at[nzmax-1, elem].set(mesh.bottom_elem_thickness[elem])
    return mesh


def init_stiff_mat_ale(mesh, partit, meshpath, g, dt, alpha, theta):
    # Get communicator information
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes

    if mype == 0:
        print("____________________________________________________________")
        print(" --> initialise ssh operator using unperturbed ocean depth")

    # Initialize and pre-allocate ssh_stiff matrix
    mesh.ssh_stiff.dim = mesh.nod2D
    mesh.ssh_stiff.rowptr = jnp.zeros(partit.myDim_nod2D + 1, dtype=jnp.int32)
    mesh.ssh_stiff.rowptr_loc = jnp.zeros(partit.myDim_nod2D + 1, dtype=jnp.int32)
    mesh.ssh_stiff.rowptr = mesh.ssh_stiff.rowptr.at[0].set(0)

    n_num = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D, dtype=jnp.int32)
    n_pos = jnp.zeros((12, partit.myDim_nod2D), dtype=jnp.int32)

    # Neighbourhood information
    for n in range(partit.myDim_nod2D):
        n_num = n_num.at[n].set(1)
        n_pos = n_pos.at[0, n].set(n)

    for n in range(partit.myDim_edge2D):
        n1 = mesh.edges[0, n]
        n2 = mesh.edges[1, n]
        if n1 < partit.myDim_nod2D:
            n_pos = n_pos.at[n_num[n1], n1].set(n2)
            n_num = n_num.at[n1].add(1)

        if n2 < partit.myDim_nod2D:
            n_pos = n_pos.at[n_num[n2], n2].set(n1)
            n_num = n_num.at[n2].add(1)


    # Fill up reduced row vector
    for n in range(partit.myDim_nod2D):
        mesh.ssh_stiff.rowptr = mesh.ssh_stiff.rowptr.at[n + 1].set(
            mesh.ssh_stiff.rowptr[n] + n_num[n]
        )

    # Calculate the number of nonzero entries
    mesh.ssh_stiff.nza = mesh.ssh_stiff.rowptr[partit.myDim_nod2D]
    print("mesh.ssh_stiff.nza=", mype, mesh.ssh_stiff.nza)
    # Allocate column and value arrays of the sparse matrix
    mesh.ssh_stiff.colind = jnp.zeros(mesh.ssh_stiff.nza, dtype=jnp.int32)
    mesh.ssh_stiff.colind_loc = jnp.zeros(mesh.ssh_stiff.nza, dtype=jnp.int32)
    mesh.ssh_stiff.values = jnp.zeros(mesh.ssh_stiff.nza)

    # Fill sparse matrix column index
    for n in range(partit.myDim_nod2D):
        nini = mesh.ssh_stiff.rowptr[n]
        nend = mesh.ssh_stiff.rowptr[n + 1]
        mesh.ssh_stiff.colind = mesh.ssh_stiff.colind.at[nini:nend].set(n_pos[:n_num[n], n])
    mesh.ssh_stiff.colind_loc = mesh.ssh_stiff.colind
    mesh.ssh_stiff.rowptr_loc = mesh.ssh_stiff.rowptr
    print("mesh.ssh_stiff.colind.sum()=", mype, jnp.sum(mesh.ssh_stiff.colind))
    # Stiffness matrix calculations
    factor = g * dt * alpha * theta
    # Loop over edges
    if (mype==0):
        print("zbar_e_bot", jnp.min(mesh.zbar_e_bot), jnp.max(mesh.zbar_e_bot))
        print("zbar_e_srf", jnp.min(mesh.zbar_e_srf), jnp.max(mesh.zbar_e_srf))


    for ed in range(partit.myDim_edge2D):
        el = mesh.edge_tri[:, ed]
        for i in range(2):
            if el[i] < 0:
               continue
            elnodes = mesh.elem2D[:, el[i]]
            fy = (mesh.zbar_e_bot[el[i]] - mesh.zbar_e_srf[el[i]]) * (
                    jnp.dot(
                        mesh.gradient_sca[:3, el[i]], mesh.edge_cross_dxdy[2 * (i+1)-1, ed]
                    )
                    - jnp.dot(
                mesh.gradient_sca[3:6, el[i]], mesh.edge_cross_dxdy[2 * (i+1) - 2, ed]
            )
            )
            if ((mype==0) & (ed==10)):
                print("fy:", i, (mesh.zbar_e_bot[el[i]] - mesh.zbar_e_srf[el[i]]), fy, mesh.gradient_sca[:, el[i]], ":", mesh.edge_cross_dxdy[:, ed])
                
            if i == 1:
                fy = -fy

            row = mesh.edges[0, ed]
            if row < partit.myDim_nod2D:
                for n in range(mesh.ssh_stiff.rowptr[row], mesh.ssh_stiff.rowptr[row + 1]):
                    n_num = n_num.at[mesh.ssh_stiff.colind[n]].set(n)
                npos = n_num[elnodes]
                mesh.ssh_stiff.values = mesh.ssh_stiff.values.at[npos].add(fy * factor)

            row = mesh.edges[1, ed]
            if row < partit.myDim_nod2D:
                for n in range(mesh.ssh_stiff.rowptr[row], mesh.ssh_stiff.rowptr[row + 1]):
                    n_num = n_num.at[mesh.ssh_stiff.colind[n]].set(n)
                npos = n_num[elnodes]
                mesh.ssh_stiff.values = mesh.ssh_stiff.values.at[npos].add(-fy * factor)
    # Mass matrix part
    for row in range(partit.myDim_nod2D):
        if mesh.ulevels_nod2D[row] > 1:
            continue
        offset = mesh.ssh_stiff.rowptr[row]
        mesh.ssh_stiff.values = mesh.ssh_stiff.values.at[offset].add(
            mesh.areasvol[mesh.ulevels_nod2D[row], row] / dt
        )
    # MPI communications
    # Convert JAX arrays to NumPy arrays for MPI communication
    pnza = np.zeros(npes, dtype='int32')  # Use explicit NumPy dtype
    rpnza = np.zeros(npes, dtype='int32')

    pnza[mype] = int(mesh.ssh_stiff.nza)  # Ensure it's an integer
    comm.Barrier()
    comm.Allreduce(pnza, rpnza, op=MPI.SUM)
    rpnza = jnp.array(rpnza)

    offset = jnp.sum(rpnza[:mype]) if mype != 0 else 0
    mesh.ssh_stiff.rowptr = mesh.ssh_stiff.rowptr + offset

    mesh.ssh_stiff.nza = jnp.sum(rpnza)

    # Convert local to global indices
    for n in range(mesh.ssh_stiff.rowptr[partit.myDim_nod2D] - mesh.ssh_stiff.rowptr[0]):
        mesh.ssh_stiff.colind = mesh.ssh_stiff.colind.at[n].set(
            partit.myList_nod2D[mesh.ssh_stiff.colind[n]]-1
        )

    mapping = np.zeros(mesh.nod2D, dtype=jnp.int32)
    dist_mesh_dir = meshpath + 'dist_' + str(partit.npes) + '/'
    file_name=dist_mesh_dir.strip() + '/rpart.out'

    if mype == 0:
        print(f"     > in stiff_mat_ale, reading {file_name}")
        with open(file_name, "r") as file:
            n = int(file.readline())
            line = file.readline()
            mapping_part1 = list(map(int, line.split()))
            mapping       = np.array([int(file.readline().split()[0])-1 for _ in range(mesh.nod2D)], dtype='int32')

    comm.Bcast(mapping, root=0)
    mapping = jnp.array(mapping)
    # Update column indices to be global
    for n in range(mesh.ssh_stiff.rowptr[partit.myDim_nod2D] - mesh.ssh_stiff.rowptr[0]):
        mesh.ssh_stiff.colind = mesh.ssh_stiff.colind.at[n].set(mapping[mesh.ssh_stiff.colind[n]])

    return mesh

def test_divergence_core(mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
                         eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
                         edges, edge_tri, edge_cross_dxdy, cyclic_length):
    """
    Core function for test_divergence without MPI dependencies.
    """
    # Allocate arrays
    ssh_rhs = jnp.zeros(myDim_nod2D + eDim_nod2D)
    velx = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)
    vely = jnp.zeros(myDim_elem2D + eDim_elem2D + eXDim_elem2D)

    # Initialize `velx` and `vely` based on element centers
    for i in range(myDim_elem2D):
        velx = velx.at[i].set(elem_center(i, elem2D, coord_nod2D, cyclic_length)[0])
        vely = vely.at[i].set(elem_center(i, elem2D, coord_nod2D, cyclic_length)[1])

    # Initialize SSH right-hand side to zero
    ssh_rhs = ssh_rhs.at[:].set(0.0)

    # Main computation loop
    for n in range(1):
        for ed in range(myDim_edge2D):
            # Nodes and elements for this edge
            enodes = edges[:, ed]
            el = edge_tri[:, ed]

            # Compute flux perpendicular to the edge from element el(1)
            deltaX1 = edge_cross_dxdy[0, ed]
            deltaY1 = edge_cross_dxdy[1, ed]
            c1 = vely[el[0]] * deltaX1 - velx[el[0]] * deltaY1

            # Using jnp.where to handle the conditional logic
            deltaX2 = jnp.where(el[1] > 0, edge_cross_dxdy[2, ed], 0.0)
            deltaY2 = jnp.where(el[1] > 0, edge_cross_dxdy[3, ed], 0.0)
            c2 = jnp.where(el[1] > 0, -(vely[el[1]] * deltaX2 + velx[el[1]] * deltaY2), 0.0)

            # Update ssh_rhs for each node in enodes
            flux_contribution = c1 + c2
            ssh_rhs = ssh_rhs.at[enodes[0]].add(flux_contribution)
            ssh_rhs = ssh_rhs.at[enodes[1]].add(-flux_contribution)

    # Compute min, max, and sum for debug output
    minval = jnp.min(ssh_rhs)
    maxval = jnp.max(ssh_rhs)
    sumval = jnp.sum(ssh_rhs)
    return ssh_rhs, minval, maxval, sumval
# Apply jax.jit as a function with static arguments
test_divergence_core = jax.jit(test_divergence_core, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 13))
def test_divergence(mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
                    eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
                    edges, edge_tri, edge_cross_dxdy, cyclic_length):
    return test_divergence_core(
        mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
        eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
        edges, edge_tri, edge_cross_dxdy, cyclic_length
    )


@jax.jit
def compute_flux(el, deltaX1, deltaY1, velx, vely, deltaX2, deltaY2):
    # Compute fluxes for each edge based on element centers using jnp.where
    c1 = vely[el[0]] * deltaX1 - velx[el[0]] * deltaY1

    # Use jnp.where to handle the conditional
    c2 = jnp.where(
        el[1] > 0,
        -(vely[el[1]] * deltaX2 + velx[el[1]] * deltaY2),
        0.0
    )

    return c1 + c2


def test_divergence_core2(mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
                         eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
                         edges, edge_tri, edge_cross_dxdy, cyclic_length):
    # Allocate arrays
    ssh_rhs = jnp.zeros(myDim_nod2D + eDim_nod2D)

    # Initialize `velx` and `vely` using vmap to compute element centers in parallel
    velx, vely = vmap(lambda i: elem_center(i, elem2D, coord_nod2D, cyclic_length))(jnp.arange(myDim_elem2D)).T

    # Initialize SSH right-hand side to zero (already zeroed in allocation)

    # Vectorized edge loop for flux calculations
    def edge_update(ed, ssh_rhs):
        enodes = edges[:, ed]
        el = edge_tri[:, ed]

        # Unpack cross products for edge flux calculation
        deltaX1, deltaY1, deltaX2, deltaY2 = edge_cross_dxdy[:, ed]

        # Compute flux contribution for each edge
        flux_contribution = compute_flux(el, deltaX1, deltaY1, velx, vely, deltaX2, deltaY2)

        # Update ssh_rhs for each node in enodes
        ssh_rhs = ssh_rhs.at[enodes[0]].add(flux_contribution)
        ssh_rhs = ssh_rhs.at[enodes[1]].add(-flux_contribution)

        return ssh_rhs

    # Use a loop over the edges with lax.scan for better compilation speed
    from jax import lax
    ssh_rhs = lax.fori_loop(0, myDim_edge2D, edge_update, ssh_rhs)

    # Compute min, max, and sum for debug output
    minval = jnp.min(ssh_rhs)
    maxval = jnp.max(ssh_rhs)
    sumval = jnp.sum(ssh_rhs)

    return ssh_rhs, minval, maxval, sumval


# Apply jax.jit as a function with static arguments
test_divergence_core2 = jax.jit(test_divergence_core2, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 13))
def test_divergence2(mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
                    eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
                    edges, edge_tri, edge_cross_dxdy, cyclic_length):
    return test_divergence_core2(
        mype, myDim_edge2D, eDim_edge2D, myDim_elem2D, eDim_elem2D,
        eXDim_elem2D, myDim_nod2D, eDim_nod2D, elem2D, coord_nod2D,
        edges, edge_tri, edge_cross_dxdy, cyclic_length
    )
