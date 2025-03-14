import numpy as np
import jax.numpy as jnp
from mpi4py import MPI


def exchange_nod3D_vectorized(nod_array3D, partit):
    """
    Vectorized version of exchange_nod3D that reduces explicit loops
    using NumPy's vectorized operations.
    """
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
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

    # Prepare the send buffer - vectorized approach
    for n in range(sn):
        nini = com_nod2D.sptr[n] - 1
        nend = com_nod2D.sptr[n + 1] - 2
        num_nodes = nend - nini + 1
        
        # Get indices of nodes to send
        send_indices = com_nod2D.slist[nini:nend+1] - 1
        
        # Allocate buffer for this process
        s_buff_nod3D[n] = np.zeros(nl1 * num_nodes)
        
        # Reshape for easier indexing
        s_buff_reshaped = s_buff_nod3D[n].reshape(num_nodes, nl1)
        
        # Vectorized data copy - transpose to match the expected layout
        s_buff_reshaped[:, :] = nod_array3D_np[:, send_indices].T
        
        # Flatten back for MPI send
        s_buff_nod3D[n] = s_buff_reshaped.flatten()

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

    # Place received data into the appropriate positions in the original array - vectorized approach
    for n in range(rn):
        nini = com_nod2D.rptr[n] - 1
        nend = com_nod2D.rptr[n + 1] - 2
        num_nodes = nend - nini + 1
        
        # Get indices of nodes to receive
        recv_indices = com_nod2D.rlist[nini:nend+1] - 1
        
        # Reshape received buffer for easier indexing
        r_buff_reshaped = r_buff_nod3D[n].reshape(num_nodes, nl1)
        
        # Vectorized data copy - transpose to match the expected layout
        nod_array3D_np[:, recv_indices] = r_buff_reshaped.T

    # Optionally convert back to JAX array if necessary
    nod_array3D = jnp.array(nod_array3D_np)

    return nod_array3D


def exchange_nod3D_flattened(nod_array3D, partit):
    """
    Flattened version of exchange_nod3D that reduces indexing overhead
    by working with flattened arrays.
    """
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array3D to NumPy array and flatten it
    nod_array3D_np = np.array(nod_array3D, copy=True)
    nl1 = nod_array3D_np.shape[0]  # Size in the vertical dimension
    total_nodes = nod_array3D_np.shape[1]
    
    # Flatten the array for more efficient indexing
    nod_array_flat = nod_array3D_np.flatten()

    # Buffers for send and receive operations
    s_buff_nod3D = [None] * sn
    r_buff_nod3D = [None] * rn

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer using flattened indexing
    for n in range(sn):
        nini = com_nod2D.sptr[n] - 1
        nend = com_nod2D.sptr[n + 1] - 2
        num_nodes = nend - nini + 1
        
        # Allocate buffer for this process
        s_buff_nod3D[n] = np.zeros(nl1 * num_nodes)
        
        # Get indices of nodes to send
        send_indices = com_nod2D.slist[nini:nend+1] - 1
        
        # Copy data using flattened indexing
        for i, idx in enumerate(send_indices):
            start_flat = i * nl1
            start_orig = idx * nl1
            s_buff_nod3D[n][start_flat:start_flat+nl1] = nod_array_flat[start_orig:start_orig+nl1]

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

    # Place received data into the appropriate positions in the original array using flattened indexing
    for n in range(rn):
        nini = com_nod2D.rptr[n] - 1
        nend = com_nod2D.rptr[n + 1] - 2
        
        # Get indices of nodes to receive
        recv_indices = com_nod2D.rlist[nini:nend+1] - 1
        
        # Copy data using flattened indexing
        for i, idx in enumerate(recv_indices):
            start_flat = i * nl1
            start_orig = idx * nl1
            nod_array_flat[start_orig:start_orig+nl1] = r_buff_nod3D[n][start_flat:start_flat+nl1]

    # Reshape back to original shape
    nod_array3D_np = nod_array_flat.reshape(nl1, total_nodes)

    # Optionally convert back to JAX array if necessary
    nod_array3D = jnp.array(nod_array3D_np)

    return nod_array3D


def exchange_nod3D_buffered(nod_array3D, partit):
    """
    Buffered version of exchange_nod3D that pre-allocates and reuses buffers
    to reduce memory allocations.
    """
    comm = partit.MPI_COMM_FESOM
    mype = partit.mype
    com_nod2D = partit.com_nod2D

    # Get the number of send/receive processes
    sn = com_nod2D.sPEnum
    rn = com_nod2D.rPEnum

    # Convert nod_array3D to NumPy array if necessary (to ensure it's writable)
    nod_array3D_np = np.array(nod_array3D, copy=True)
    nl1 = nod_array3D_np.shape[0]  # Size in the vertical dimension

    # Pre-allocate buffers for all send/receive operations
    # This avoids memory allocations during the exchange
    s_buff_sizes = np.zeros(sn, dtype=np.int32)
    r_buff_sizes = np.zeros(rn, dtype=np.int32)
    
    # Calculate buffer sizes
    for n in range(sn):
        nini = com_nod2D.sptr[n]
        nend = com_nod2D.sptr[n + 1]
        s_buff_sizes[n] = (nend - nini) * nl1
    
    for n in range(rn):
        nini = com_nod2D.rptr[n]
        nend = com_nod2D.rptr[n + 1]
        r_buff_sizes[n] = (nend - nini) * nl1
    
    # Allocate all buffers at once
    s_buff_nod3D = [np.zeros(size) for size in s_buff_sizes]
    r_buff_nod3D = [np.zeros(size) for size in r_buff_sizes]

    # Store send/receive requests
    sreq = []
    rreq = []

    # Prepare the send buffer with optimized indexing
    for n in range(sn):
        nini = com_nod2D.sptr[n] - 1
        nend = com_nod2D.sptr[n + 1] - 2
        
        # Get indices of nodes to send
        send_indices = com_nod2D.slist[nini:nend+1] - 1
        
        # Fill buffer with optimized indexing
        idx = 0
        for i in range(len(send_indices)):
            node_idx = send_indices[i]
            for z in range(nl1):
                s_buff_nod3D[n][idx] = nod_array3D_np[z, node_idx]
                idx += 1

    # Start non-blocking MPI send and receive operations simultaneously
    # to overlap communication
    for n in range(sn):
        dest = com_nod2D.sPE[n]
        req = comm.Isend(s_buff_nod3D[n], dest=dest, tag=mype)
        sreq.append(req)

    for n in range(rn):
        source = com_nod2D.rPE[n]
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
        
        # Get indices of nodes to receive
        recv_indices = com_nod2D.rlist[nini:nend+1] - 1
        
        # Fill array with optimized indexing
        idx = 0
        for i in range(len(recv_indices)):
            node_idx = recv_indices[i]
            for z in range(nl1):
                nod_array3D_np[z, node_idx] = r_buff_nod3D[n][idx]
                idx += 1

    # Optionally convert back to JAX array if necessary
    nod_array3D = jnp.array(nod_array3D_np)

    return nod_array3D
