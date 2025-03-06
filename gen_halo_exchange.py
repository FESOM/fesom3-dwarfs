import numpy as np
import jax
import jax.numpy as jnp
from mpi4py import MPI

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

def exchange_elem3D(elem_array3D, partit):
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    npes = partit.npes
    com_elem2D = partit.com_elem2D

    # Get the number of send/receive processes
    sn = com_elem2D.sPEnum
    rn = com_elem2D.rPEnum

    # Convert elem_array3D to NumPy array if necessary (to ensure it's writable)
    elem_array3D_np = np.array(elem_array3D, copy=True)
    nl1 = elem_array3D_np.shape[0]  # Size in the vertical dimension

    # Buffers for send and receive operations
    s_buff_elem3D = [None] * sn
    r_buff_elem3D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer
    for n in range(sn):
        nini = com_elem2D.sptr[n] - 1
        nend = com_elem2D.sptr[n + 1] - 2
        nc = 0
        s_buff_elem3D[n] = np.zeros((nl1 * (nend - nini + 1)))
        for nh in range(nini, nend + 1):
            for nz in range(nl1):
                s_buff_elem3D[n][nc] = elem_array3D_np[nz, com_elem2D.slist[nh] - 1]
                nc += 1

    # Non-blocking MPI send
    for n in range(sn):
        dest = com_elem2D.sPE[n]
        nini = com_elem2D.sptr[n]
        offset = (com_elem2D.sptr[n + 1] - nini) * nl1
        req = comm.Isend(s_buff_elem3D[n], dest=dest, tag=mype)
        sreq.append(req)

    # Non-blocking MPI receive
    for n in range(rn):
        source = com_elem2D.rPE[n]
        nini = com_elem2D.rptr[n]
        offset = (com_elem2D.rptr[n + 1] - nini) * nl1
        r_buff_elem3D[n] = np.zeros(offset)
        req = comm.Irecv(r_buff_elem3D[n], source=source, tag=source)
        rreq.append(req)

    # Wait for all send operations to complete
    MPI.Request.Waitall(sreq)

    # Wait for all receive operations to complete
    MPI.Request.Waitall(rreq)

    # Place received data into the appropriate positions in the original array
    for n in range(rn):
        nini = com_elem2D.rptr[n] - 1
        nend = com_elem2D.rptr[n + 1] - 2
        nc = 0
        for nh in range(nini, nend + 1):
            for nz in range(nl1):
                elem_array3D_np[nz, com_elem2D.rlist[nh] - 1] = r_buff_elem3D[n][nc]
                nc += 1

    # Optionally convert back to JAX array if necessary
    elem_array3D = jnp.array(elem_array3D_np)

    return elem_array3D