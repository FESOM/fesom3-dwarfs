# mesh_partit_types.py

from dataclasses import dataclass, field
import jax.numpy as jnp

# SparseMatrix class
@dataclass
class SparseMatrix:
    nza: int=0
    dim: int=0
    values: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    colind: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rowptr: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    colind_loc: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rowptr_loc: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    pr_values: jnp.ndarray = field(default_factory=lambda: jnp.array([]))

# Mesh class (T_MESH equivalent)
@dataclass
class Mesh:
    nod2D: int = 0
    ocean_area: float = 0.0
    ocean_areawithcav: float = 0.0
    coord_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    geo_coord_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    edge2D: int = 0
    edge2D_in: int = 0
    elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    edges: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    edge_tri: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    elem_edges: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    elem_area: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    edge_dxdy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    edge_cross_dxdy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    elem_cos: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    metric_factor: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    elem_neighbors: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    nod_in_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32))
    x_corners: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    y_corners: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    nod_in_elem2D_num: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    depth: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    gradient_vec: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    gradient_sca: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    nl: int = 0
    zbar: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    Z: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    elem_depth: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    ulevels: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    ulevels_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    nlevels: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    area: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32))
    mesh_resolution: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    cavity_flag_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    cavity_flag_e: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    ssh_stiff: SparseMatrix = field(default_factory=SparseMatrix)
    coriolis: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))
    coriolis_node: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float32))

# CommunicationStruct class (com_struct equivalent)
@dataclass
class CommunicationStruct:
    rPEnum: int = 0
    rPE: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    rptr: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    rlist: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    sPEnum: int = 0
    sPE: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    sptr: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    slist: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))

# Partitioning class (T_PARTIT equivalent)
@dataclass
class Partitioning:
    MPI_COMM_FESOM_IB: int = 0
    MPIERR_IB: int = 0
    com_nod2D: CommunicationStruct = field(default_factory=CommunicationStruct)
    com_elem2D: CommunicationStruct = field(default_factory=CommunicationStruct)
    com_elem2D_full: CommunicationStruct = field(default_factory=CommunicationStruct)
    npes: int = 0
    mype: int = 0
    part: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    myDim_nod2D: int = 0
    eDim_nod2D: int = 0
    myList_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    myDim_elem2D: int = 0
    eDim_elem2D: int = 0
    eXDim_elem2D: int = 0
    myList_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    myDim_edge2D: int = 0
    eDim_edge2D: int = 0
    myList_edge2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    pe_status: int = 0
    MPI_COMM_FESOM: int = 0
    MPI_COMM_WORLD: int = 0
    remPtr_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    remList_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    remPtr_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    remList_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))

