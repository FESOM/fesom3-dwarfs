from mpi4py import MPI
import pickle

def save_data(data, filename_prefix, mype):
    """
    Save data to a file specific to each MPI process.
    """
    filename = f"{filename_prefix}_rank_{mype}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Rank {mype}: Data saved to {filename}")

def load_data(filename_prefix, mype):
    """
    Load data from a file specific to each MPI process.
    """
    filename = f"{filename_prefix}_rank_{mype}.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Rank {mype}: Data loaded from {filename}")
    return data
