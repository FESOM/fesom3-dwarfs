import os
import sys
import time
import pickle
import numpy as np

# Set up environment variables for JAX before importing it
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

# Now import JAX-related modules
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from mpi4py import MPI
# These classes needed for unpickling mesh and partition data
from data_types import Mesh, Partitioning  # noqa: F401
from gen_halo_exchange import exchange_nod3D, exchange_elem3D
from gen_halo_exchange_optimized_nod3D import (
    exchange_nod3D_vectorized,
    exchange_nod3D_flattened,
    exchange_nod3D_buffered
)
from gen_halo_exchange_optimized import (
    exchange_elem3D_vectorized,
    exchange_elem3D_flattened,
    exchange_elem3D_buffered
)


def load_data(prefix, rank, size):
    """Load mesh or partitioning data from pickle files"""
    filename = f"preprocessed_mesh/{prefix}_rank_{rank}_of_{size}.pkl"
    
    # Check if file exists first
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist.")
        print(f"Current directory: {os.getcwd()}")
        print("Available files in preprocessed_mesh/:")
        try:
            for f in os.listdir("preprocessed_mesh/"):
                print(f"  {f}")
        except Exception as e:
            print(f"  Could not list directory: {e}")
        sys.exit(1)
    
    try:
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            
        # Convert dictionary to appropriate object
        if prefix == "partit":
            data = Partitioning()
            # Copy all attributes from the dictionary to the object
            for key, value in data_dict.items():
                if isinstance(value, dict) and key == "com_nod2D":
                    # Handle nested CommunicationStruct objects
                    from data_types import CommunicationStruct
                    comm_struct = CommunicationStruct()
                    for k, v in value.items():
                        setattr(comm_struct, k, v)
                    setattr(data, key, comm_struct)
                elif isinstance(value, dict) and key == "com_elem2D":
                    # Handle nested CommunicationStruct objects
                    from data_types import CommunicationStruct
                    comm_struct = CommunicationStruct()
                    for k, v in value.items():
                        setattr(comm_struct, k, v)
                    setattr(data, key, comm_struct)
                else:
                    setattr(data, key, value)
        else:  # Mesh object
            data = Mesh()
            # Copy all attributes from the dictionary to the object
            for key, value in data_dict.items():
                setattr(data, key, value)
                
        print(f"Successfully loaded and converted {prefix} data")
        return data
    except Exception as e:
        print(f"Error loading/converting {filename}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def benchmark_function(func, array, partit, num_runs=50):
    """Benchmark a single exchange function"""
    comm = partit.MPI_COMM_FESOM
    rank = partit.mype
    
    # Store timing results
    times = []
    
    # Warm-up run
    _ = func(array, partit)
    
    # Synchronize before starting benchmark
    comm.Barrier()
    
    # Run benchmark
    for i in range(num_runs):
        # Synchronize before each run
        comm.Barrier()
        
        # Reset array to rank+1 values for clear verification
        if isinstance(array, np.ndarray):
            array.fill(rank + 1)
        else:
            array = jnp.ones_like(array) * (rank + 1)
        
        # Time the exchange
        start_time = time.time()
        result = func(array, partit)
        end_time = time.time()
        
        # Store timing
        times.append(end_time - start_time)
        
        # Verify exchange worked by checking if values changed at boundary
        if i == 0:  # Only print verification for first run
            num_changed = jnp.sum(result != (rank + 1))
            print(f"Rank {rank}: {func.__name__} changed {num_changed} values")
    
    # Calculate statistics
    mean_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    print(f"Rank {rank}: {func.__name__} timing (seconds):")
    print(f"  Mean: {mean_time:.6f}")
    print(f"  Min: {min_time:.6f}")
    print(f"  Max: {max_time:.6f}")
    print(f"  Median: {median_time:.6f}")
    
    return times


def run_optimized_benchmark(comm, mesh, partit, num_runs=50):
    """Run benchmarks comparing original and optimized exchange functions"""
    rank = partit.mype
    
    # Collect mesh information
    local_info = {
        "rank": rank,
        "nl": mesh.nl,
        "myDim_nod2D": partit.myDim_nod2D,
        "eDim_nod2D": partit.eDim_nod2D,
        "total_nodes": partit.myDim_nod2D + partit.eDim_nod2D,
        "myDim_elem2D": partit.myDim_elem2D,
        "eDim_elem2D": partit.eDim_elem2D,
        "total_elems": partit.myDim_elem2D + partit.eDim_elem2D
    }
    
    # Use MPI reduction to get global node and element counts
    global_total_nodes = comm.allreduce(partit.myDim_nod2D, op=MPI.SUM)
    global_total_elems = comm.allreduce(partit.myDim_elem2D, op=MPI.SUM)
    all_info = comm.gather(local_info, root=0)
    
    # Print mesh information
    if rank == 0:
        print("\n----- MESH INFORMATION -----")
        print(f"Total number of vertical levels: {mesh.nl}")
        print(f"Total number of nodes in mesh: {global_total_nodes}")
        print(f"Total number of elements in mesh: {global_total_elems}")
        print("\n----- PER-RANK DISTRIBUTION -----")
        
        # Create a table for clarity
        print(f"{'Rank':<6} {'Nodes':<10} {'Ghost Nodes':<15} {'Elements':<10} {'Ghost Elements':<15}")
        print("-" * 60)
        
        for info in all_info:
            r = info["rank"]
            print(f"{r:<6} {info['myDim_nod2D']:<10} {info['eDim_nod2D']:<15} "
                  f"{info['myDim_elem2D']:<10} {info['eDim_elem2D']:<15}")
        
        print("\n----- BENCHMARK STARTING -----")
    
    # Make sure all ranks wait for rank 0 to print the information
    comm.Barrier()
    
    # Individual rank info
    print(f"Rank {rank}: Starting with {partit.myDim_nod2D} nodes and {partit.myDim_elem2D} elements")
    
    # Create test arrays
    # 3D arrays
    total_nodes = partit.myDim_nod2D + partit.eDim_nod2D
    total_elems = partit.myDim_elem2D + partit.eDim_elem2D
    nod_array3D = jnp.ones((mesh.nl, total_nodes)) * (rank + 1)
    elem_array3D = jnp.ones((mesh.nl-1, total_elems)) * (rank + 1)
    
    # Dictionary to store all timing results
    all_times = {}
    
    # Benchmark original exchange_nod3D
    print(f"\nRank {rank}: Benchmarking original exchange_nod3D...")
    all_times["nod3D_orig"] = benchmark_function(
        exchange_nod3D, nod_array3D, partit, num_runs
    )
    
    # Benchmark vectorized exchange_nod3D
    print(f"\nRank {rank}: Benchmarking vectorized exchange_nod3D...")
    all_times["nod3D_vec"] = benchmark_function(
        exchange_nod3D_vectorized, nod_array3D, partit, num_runs
    )
    
    # Benchmark flattened exchange_nod3D
    print(f"\nRank {rank}: Benchmarking flattened exchange_nod3D...")
    all_times["nod3D_flat"] = benchmark_function(
        exchange_nod3D_flattened, nod_array3D, partit, num_runs
    )
    
    # Benchmark buffered exchange_nod3D
    print(f"\nRank {rank}: Benchmarking buffered exchange_nod3D...")
    all_times["nod3D_buf"] = benchmark_function(
        exchange_nod3D_buffered, nod_array3D, partit, num_runs
    )
    
    # Benchmark original exchange_elem3D
    print(f"\nRank {rank}: Benchmarking original exchange_elem3D...")
    all_times["elem3D_orig"] = benchmark_function(
        exchange_elem3D, elem_array3D, partit, num_runs
    )
    
    # Benchmark vectorized exchange_elem3D
    print(f"\nRank {rank}: Benchmarking vectorized exchange_elem3D...")
    all_times["elem3D_vec"] = benchmark_function(
        exchange_elem3D_vectorized, elem_array3D, partit, num_runs
    )
    
    # Benchmark flattened exchange_elem3D
    print(f"\nRank {rank}: Benchmarking flattened exchange_elem3D...")
    all_times["elem3D_flat"] = benchmark_function(
        exchange_elem3D_flattened, elem_array3D, partit, num_runs
    )
    
    # Benchmark buffered exchange_elem3D
    print(f"\nRank {rank}: Benchmarking buffered exchange_elem3D...")
    all_times["elem3D_buf"] = benchmark_function(
        exchange_elem3D_buffered, elem_array3D, partit, num_runs
    )
    
    # Calculate data sizes
    nod3D_size = mesh.nl * total_nodes * 8  # 8 bytes per double
    elem3D_size = (mesh.nl-1) * total_elems * 8  # 8 bytes per double
    
    # Calculate mean times
    mean_times = {k: np.mean(v) for k, v in all_times.items()}
    
    # Print comparison if rank 0
    if rank == 0:
        print("\n----- OPTIMIZATION COMPARISON -----")
        print(f"{'Function':<20} {'Mean Time (s)':<15} {'Speedup':<10}")
        print("-" * 45)
        
        # Nodal 3D exchanges
        nod3D_orig_mean = mean_times["nod3D_orig"]
        print(f"{'exchange_nod3D':<20} {nod3D_orig_mean:<15.6f} {1.0:<10.2f}")
        
        nod3D_vec_mean = mean_times["nod3D_vec"]
        print(f"{'nod3D_vectorized':<20} {nod3D_vec_mean:<15.6f} {nod3D_orig_mean/nod3D_vec_mean:<10.2f}")
        
        nod3D_flat_mean = mean_times["nod3D_flat"]
        print(f"{'nod3D_flattened':<20} {nod3D_flat_mean:<15.6f} {nod3D_orig_mean/nod3D_flat_mean:<10.2f}")
        
        nod3D_buf_mean = mean_times["nod3D_buf"]
        print(f"{'nod3D_buffered':<20} {nod3D_buf_mean:<15.6f} {nod3D_orig_mean/nod3D_buf_mean:<10.2f}")
        
        print("\n")
        
        # Element 3D exchanges
        elem3D_orig_mean = mean_times["elem3D_orig"]
        print(f"{'exchange_elem3D':<20} {elem3D_orig_mean:<15.6f} {1.0:<10.2f}")
        
        elem3D_vec_mean = mean_times["elem3D_vec"]
        print(f"{'elem3D_vectorized':<20} {elem3D_vec_mean:<15.6f} {elem3D_orig_mean/elem3D_vec_mean:<10.2f}")
        
        elem3D_flat_mean = mean_times["elem3D_flat"]
        print(f"{'elem3D_flattened':<20} {elem3D_flat_mean:<15.6f} {elem3D_orig_mean/elem3D_flat_mean:<10.2f}")
        
        elem3D_buf_mean = mean_times["elem3D_buf"]
        print(f"{'elem3D_buffered':<20} {elem3D_buf_mean:<15.6f} {elem3D_orig_mean/elem3D_buf_mean:<10.2f}")
        
        # Find the best performing versions
        best_nod3D = min(
            ["nod3D_orig", "nod3D_vec", "nod3D_flat", "nod3D_buf"],
            key=lambda k: mean_times[k]
        )
        best_elem3D = min(
            ["elem3D_orig", "elem3D_vec", "elem3D_flat", "elem3D_buf"],
            key=lambda k: mean_times[k]
        )
        
        print("\n----- BEST PERFORMING VERSIONS -----")
        print(f"Best nod3D version: {best_nod3D} - {mean_times[best_nod3D]:.6f} seconds")
        print(f"Best elem3D version: {best_elem3D} - {mean_times[best_elem3D]:.6f} seconds")
    
    # Gather all timing results
    all_rank_times = comm.gather({
        "rank": rank,
        "times": all_times
    }, root=0)
    
    return all_times


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Ensure we have either 2 or 4 MPI tasks
    if size not in [2, 4]:
        if rank == 0:
            print("This benchmark requires either 2 or 4 MPI tasks.")
        MPI.Finalize()
        sys.exit(1)
    
    # Load mesh and partitioning data with the correct size
    partit = load_data("partit", rank, size)
    mesh = load_data("mesh", rank, size)
    
    # Set MPI communicator
    partit.MPI_COMM_FESOM = comm
    
    # Run benchmark
    all_times = run_optimized_benchmark(comm, mesh, partit)
    
    # Synchronize before finishing
    comm.Barrier()
    
    if rank == 0:
        print("\nBenchmark completed!")
    
    MPI.Finalize()


if __name__ == "__main__":
    main()
