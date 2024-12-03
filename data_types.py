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
    coord_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    geo_coord_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    edge2D: int = 0
    edge2D_in: int = 0
    elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    edges: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    edge_tri: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    elem_edges: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    elem_area: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    edge_dxdy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    edge_cross_dxdy: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    elem_cos: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    metric_factor: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    elem_neighbors: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nod_in_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    x_corners: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    y_corners: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nod_in_elem2D_num: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    depth: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    gradient_vec: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    gradient_sca: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nl: int = 0
    zbar: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    Z: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    elem_depth: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    ulevels: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    ulevels_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    nlevels: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    area: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    mesh_resolution: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cavity_flag_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cavity_flag_e: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    ssh_stiff: SparseMatrix = field(default_factory=SparseMatrix)
    coriolis: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    coriolis_node: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    hnode: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    hnode_new: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    hbar: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    hbar_old: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    helem: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    dhe: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    zbar_3d_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    Z_3d_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    Z_3d_n_ib: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
# CommunicationStruct class (com_struct equivalent)
@dataclass
class CommunicationStruct:
    rPEnum: int = 0
    rPE: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    rptr: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    rlist: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    sPEnum: int = 0
    sPE: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    sptr: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    slist: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))

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
    part: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    myDim_nod2D: int = 0
    eDim_nod2D: int = 0
    myList_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    myDim_elem2D: int = 0
    eDim_elem2D: int = 0
    eXDim_elem2D: int = 0
    myList_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    myDim_edge2D: int = 0
    eDim_edge2D: int = 0
    myList_edge2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    pe_status: int = 0
    MPI_COMM_FESOM: int = 0
    MPI_COMM_WORLD: int = 0
    remPtr_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    remList_nod2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    remPtr_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    remList_elem2D: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))

@dataclass
class Dynamics:
    ssh_rhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    ssh_rhs_old: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    eta_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    deta_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
#    u: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    v: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    w: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    urhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    vrhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    urhsAB: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
#    vrhsAB: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))

    UV_rhs   : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    UV_rhsAB : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0, 0)))
    uv       : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    AB_order: int = 2


@dataclass
class Dynamics2:
    ssh_rhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    ssh_rhs_old: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    eta_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    deta_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
#    u: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    v: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    w: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    urhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    vrhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
#    urhsAB: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
#    vrhsAB: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))

    U_rhs   : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    V_rhs   : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    U_rhsAB : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    V_rhsAB : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    u       : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    v       : jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    AB_order: int = 2