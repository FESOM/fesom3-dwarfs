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
from data_types import Mesh, Partitioning, Dynamics, Dynamics2
from read_write_mesh_binary import *
force_rotation = True

# Set up environment variables for OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPI4JAX_USE_CUDA_MPI"] = "1"

# Ensure JAX uses the GPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
mesh = Mesh()
# Initialize MPI
# comm = MPI.COMM_WORLD
#partit = Partitioning(npes=MPI.COMM_WORLD.Get_size(), mype=MPI.COMM_WORLD.Get_rank(), MPI_COMM_FESOM=MPI.COMM_WORLD)
npes=MPI.COMM_WORLD.Get_size()
mype=MPI.COMM_WORLD.Get_rank()
MPI_COMM_FESOM=MPI.COMM_WORLD
partit = Partitioning(npes=npes, mype=mype, MPI_COMM_FESOM=MPI_COMM_FESOM)
dynamics =Dynamics()
dynamics2=Dynamics2()
# Ensure we have exactly 4 MPI tasks
assert npes == 4, "This example requires exactly 4 MPI tasks."
meshpath = '/home/dsidoren/myapps/test/pi/'
from read_mesh_and_partition import *
set_mesh_transform_matrix(50.*jnp.pi/180., 15.*jnp.pi/180., -90.*jnp.pi/180.)

cyclic_length=2.*jnp.pi
r_earth=6367500.0
g=9.8
dt=1800.
alpha=1.0
theta=1.0
do_read_mesh_ascii=False

import pickle
if do_read_mesh_ascii:
    mesh, partit=read_mesh_and_partition(mesh, partit, meshpath, force_rotation=True)
    mesh, partit=test_tri(mesh, partit, cyclic_length)
    mesh, partit=load_edges(mesh, partit, meshpath)
    mesh, partit=find_neighbors(mesh, partit)
    mesh, partit=find_levels(mesh, partit, meshpath)
    mesh, partit=find_levels_min_e2n(mesh, partit)
    mesh, partit=mesh_areas(mesh, partit, cartesian=False, cyclic_length=cyclic_length, r_earth=r_earth)
    mesh_auxiliary_arrays(mesh, partit, cartesian=False, fplane=False, cyclic_length=cyclic_length, r_earth=r_earth)
    mesh=init_ale(mesh, partit)
    mesh=init_thickness_ale(mesh, partit)
    mesh=init_stiff_mat_ale(mesh, partit, meshpath, g, dt, alpha, theta)
    partit.MPI_COMM_FESOM=0
    save_data(partit, "partit", mype)
    partit.MPI_COMM_FESOM=MPI_COMM_FESOM
    save_data(mesh, "mesh", mype)
else:
    partit = load_data("partit", mype)
    partit.MPI_COMM_FESOM = MPI_COMM_FESOM
    mesh = load_data("mesh", mype)
import time


from oce_dynamics import *

dynamics.eta_n = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics.UV_rhs = jnp.zeros((2, mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.UV_rhsAB = jnp.zeros((2, 2, mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.uv = jnp.zeros((2, mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))

dynamics2.eta_n = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics2.U_rhs = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1))
dynamics2.V_rhs = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1))
dynamics2.U_rhsAB = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1, 2))
dynamics2.V_rhsAB = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1, 2))
dynamics2.u = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1))
dynamics2.v = jnp.zeros((partit.myDim_elem2D + partit.eDim_elem2D, mesh.nl-1))

compute_vel_rhs_opt2_jit(dynamics2.u, dynamics2.v, dynamics2.U_rhs, dynamics2.V_rhs, dynamics2.U_rhsAB, dynamics2.V_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
t1 = time.time()
for i in range(10):
    compute_vel_rhs_opt2_jit(dynamics2.u, dynamics2.v, dynamics2.U_rhs, dynamics2.V_rhs, dynamics2.U_rhsAB, dynamics2.V_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
t2 = time.time()
if (partit.mype) == 0: print("compute rhs optimized split (jax, lax):", t2 - t1)

#t1 = time.time()
#for i in range(10):
#    compute_vel_rhs(dynamics, partit, mesh, g, dt)
#t2 = time.time()
#if (partit.mype) == 0: print("compute rhs unoptimized (jax, lax):", t2 - t1)

t1 = time.time()
for i in range(10):
    compute_vel_rhs_opt(dynamics.uv, dynamics.UV_rhs, dynamics.UV_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
t2 = time.time()
if (partit.mype) == 0: print("compute rhs without @jit          :", t2 - t1)

print(jnp.shape(mesh.gradient_sca), jnp.shape(dynamics.eta_n[mesh.elem2D]))
Fx = jnp.sum(mesh.gradient_sca[:3, :] * dynamics.eta_n[mesh.elem2D], axis=0)
Fy = jnp.sum(mesh.gradient_sca[3:, :] * dynamics.eta_n[mesh.elem2D], axis=0)
compute_vel_rhs_opt3_jit(dynamics.uv, dynamics.UV_rhs, dynamics.UV_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)

from jax import profiler
t1 = time.time()
# Profiling block
with profiler.trace("/home/dsidoren/FESOM3/jax_profile"):
    compute_vel_rhs_opt3_jit(dynamics.uv, dynamics.UV_rhs, dynamics.UV_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
t2 = time.time()
if (partit.mype) == 0: print("compute rhs @jit second call      :", t2 - t1)

#call it first time (will be long since needs be compiled)
t1 = time.time()
ssh_rhs, minval, maxval, sumval = test_divergence2(mype=partit.mype, myDim_edge2D = partit.myDim_edge2D,
    eDim_edge2D = partit.eDim_edge2D, myDim_elem2D = partit.myDim_elem2D, eDim_elem2D = partit.eDim_elem2D,
    eXDim_elem2D = partit.eXDim_elem2D, myDim_nod2D = partit.myDim_nod2D, eDim_nod2D = partit.eDim_nod2D,
    elem2D=mesh.elem2D, coord_nod2D=mesh.coord_nod2D, edges=mesh.edges, edge_tri=mesh.edge_tri, edge_cross_dxdy=mesh.edge_cross_dxdy,
    cyclic_length=cyclic_length)
t2 = time.time()
if (partit.mype) == 0:
    print(f"div_test: {partit.mype}, minval: {minval}, maxval: {maxval}, sum: {sumval}, time: {t2 - t1}")

#call it second time (shall be fast)
t1 = time.time()
for i in range(1000):
    ssh_rhs, minval, maxval, sumval = test_divergence2(mype=partit.mype, myDim_edge2D = partit.myDim_edge2D,
        eDim_edge2D = partit.eDim_edge2D, myDim_elem2D = partit.myDim_elem2D, eDim_elem2D = partit.eDim_elem2D,
        eXDim_elem2D = partit.eXDim_elem2D, myDim_nod2D = partit.myDim_nod2D, eDim_nod2D = partit.eDim_nod2D,
        elem2D=mesh.elem2D, coord_nod2D=mesh.coord_nod2D, edges=mesh.edges, edge_tri=mesh.edge_tri, edge_cross_dxdy=mesh.edge_cross_dxdy,
        cyclic_length=cyclic_length)
t2 = time.time()
if (partit.mype) == 0:
    print(f"div_test: {partit.mype}, minval: {minval}, maxval: {maxval}, sum: {sumval}, time: {t2 - t1}")

partit.MPI_COMM_FESOM.Barrier()
MPI.Finalize()