import os
import jax
#import pytest
import jax.numpy as jnp
from mpi4py import MPI
#from mpi4jax import send, recv, bcast
from jax import random
from jax import ops
from array_interfaces import array_factory, JaxStyleNumpyArray
import numpy as np
from data_types import Mesh, Partitioning, Dynamics, SolverInfo
from read_write_mesh_binary import *
force_rotation = True

# Set up environment variables for OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MPI4JAX_USE_CUDA_MPI"] = "1"

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
# Ensure we have exactly 4 MPI tasks
assert npes == 4, "This example requires exactly 4 MPI tasks."
meshpath = '/home/dsidoren/myapps/test/pi/'
from read_mesh_and_partition import *
set_mesh_transform_matrix(50.*jnp.pi/180., 15.*jnp.pi/180., -90.*jnp.pi/180.)

cyclic_length=2.*jnp.pi
r_earth=6367500.0
g=9.81
dt=1800.
alpha=1.0
theta=1.0
do_read_mesh_ascii=False
which_ALE = 'linfs'  # Default ALE scheme as in Fortran

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

#mesh.helem = exchange_elem2D(mesh.helem, partit)
from oce_dynamics import *

mesh.helem = exchange_elem3D(mesh.helem, partit)
mesh.areasvol = exchange_nod3D(mesh.areasvol, partit)
print(partit.mype,jnp.min(mesh.helem ), jnp.max(mesh.helem))

dynamics.eta_n = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics.U_rhs = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.V_rhs = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.U_rhsAB = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D, 2))
dynamics.V_rhsAB = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D, 2))
dynamics.u = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.v = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.u_c = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.v_c = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.w = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
dynamics.w_i = jnp.zeros((mesh.nl, partit.myDim_nod2D + partit.eDim_nod2D))
dynamics.eta_n = mesh.coord_nod2D[0,:]
dynamics.water_flux = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics.ssh_rhs = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics.ssh_rhs_old = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)

stress_surf=jnp.zeros((2, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.Av=jnp.zeros((mesh.nl, partit.myDim_elem2D + partit.eDim_elem2D))
dynamics.Av = dynamics.Av.at[:,:].set(1.)
stress_surf = stress_surf.at[:,:].set(1.)
C_d=1.0#1.e-3

arr = jnp.zeros((mesh.nl-1, partit.myDim_elem2D + partit.eDim_elem2D))
arr = arr.at[:, :partit.myDim_elem2D].set(1)
print("before: ", arr.min(), arr.max())
arr=exchange_elem3D(arr, partit)
print("after : ", arr.min(), arr.max())

t1 = time.time()
U_rhs, V_rhs, U_rhsAB, V_rhsAB = compute_vel_rhs_opt_jit(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.U_rhsAB, dynamics.V_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)

t2 = time.time()

if (partit.mype==0):
    print("compilation time for compute_vel_rhs_opt_jit:", t2 - t1)

if partit.mype == 1:
    print(mesh.nlevels[8])
if partit.mype == 1:
    print("U_rhs", U_rhs[:,8])

t1 = time.time()
for i in range (5):
    dynamics.U_rhs, dynamics.V_rhs, dynamics.U_rhsAB, dynamics.V_rhsAB = compute_vel_rhs_opt_jit(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.U_rhsAB, dynamics.V_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
t2 = time.time()

if (partit.mype==0):
    print("runtime for compute_vel_rhs_opt_jit:", t2 - t1)
t1 = time.time()


if partit.mype == 1:
    print("U_rhs", U_rhs[:,8])

dynamics.u = jnp.tile(
    mesh.elem_cos[: partit.myDim_elem2D + partit.eDim_elem2D],
    (mesh.nl - 1, 1)
)
dynamics.u_c, dynamics.v_c = visc_filt_bilapl_first_jit(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.u_c, dynamics.v_c,
                                    mesh.ulevels, mesh.nlevels, mesh.elem_area, mesh.edge_tri,
                                    dynamics.visc_gamma0, dynamics.visc_gamma1, dynamics.visc_gamma2,
                                    dt, partit.myDim_elem2D, partit.eDim_elem2D, partit.myDim_edge2D, partit.eDim_edge2D)
dynamics.u_c = exchange_elem3D(dynamics.u_c, partit)  # needs to be taken out of jax
dynamics.v_c = exchange_elem3D(dynamics.v_c, partit)  # needs to be taken out of jax
# Call the second function (handles external function calls and completes computation)
dynamics.U_rhs, dynamics.V_rhs, U_c, V_c = visc_filt_bilapl_second_jit(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.u_c,
                                                 dynamics.v_c, mesh.ulevels, mesh.nlevels, mesh.elem_area, mesh.edge_tri,
                                                 partit.myDim_edge2D, partit.eDim_edge2D)
                                                 
t2 = time.time()


if (partit.mype==0):
    print("runtime for visc_filt_bilapl_jit:", t2 - t1)


if partit.mype == 1:
    print("............................................................")
    print("nlevels:", mesh.ulevels[8], mesh.nlevels[8])
    print("helem:",   mesh.helem[:, 8])
    print("zbar_e_bot:",   mesh.zbar_e_bot[8])
    


if partit.mype == 1:
    print("U_rhs 1", dynamics.U_rhs[:,8])

if partit.mype == 1:
    print("U 1", dynamics.u[:,8])


dynamics.U_rhs, dynamics.V_rhs = impl_vert_visc_ale_opt_jit(
    dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.w_i,
    stress_surf, dynamics.Av, mesh.elem_area, mesh.elem2D, mesh.ulevels, mesh.nlevels,
    mesh.zbar_e_bot, mesh.helem, C_d, partit.myDim_elem2D, dt)


if partit.mype == 1:
    print("U_rhs 2", dynamics.U_rhs[:,8])
print("U_rhs total sum 2:", partit.mype, jnp.sum(dynamics.U_rhs[:,:partit.myDim_elem2D]))

if partit.mype == 1:
    print(mesh.ulevels[8], mesh.nlevels[8])

dynamics.ssh_rhs = compute_ssh_rhs_ale_jit(
    dynamics.u, dynamics.v,  # U and V velocities
    dynamics.U_rhs, dynamics.V_rhs,  # RHS terms
    dynamics.ssh_rhs, dynamics.ssh_rhs_old,  # SSH terms
    dynamics.water_flux, alpha,  # Water flux and alpha parameter
    mesh.edges, mesh.edge_tri, mesh.edge_cross_dxdy,  # Mesh geometry
    mesh.ulevels, mesh.nlevels, mesh.helem, mesh.areasvol,  # Level and area info
    partit.myDim_nod2D, partit.myDim_edge2D,  # Partition info
    which_ALE  # ALE configuration
)

dynamics.ssh_rhs = exchange_nod2D(dynamics.ssh_rhs, partit)



print("ssh_rhs_sum=", partit.mype, jnp.sum(dynamics.ssh_rhs))
#print("edge_cross_dxdy=", partit.mype, jnp.sum(mesh.edge_cross_dxdy[2,:]))

#print(jnp.min(mesh.edges[0,:]), jnp.min(mesh.edges[1, :]))
#print(jnp.min(mesh.edge_tri[0,:]), jnp.min(mesh.edge_tri[1, :]))
#print(partit.mype, partit.myDim_nod2D, partit.myDim_edge2D)

# Initialize preconditioner
solverinfo = SolverInfo(partit.myDim_nod2D, partit.eDim_nod2D)
rr, zz, pp, App = ssh_solve_preconditioner_jit(solverinfo, partit, mesh)

# Initialize and solve SSH equation
dynamics.d_eta = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
dynamics.d_eta = ssh_solve_cg_jit(dynamics.ssh_rhs, dynamics.d_eta, solverinfo, mesh, partit)
print("d_eta_sum=", partit.mype, jnp.sum(dynamics.d_eta))

partit.MPI_COMM_FESOM.Barrier()
MPI.Finalize()
