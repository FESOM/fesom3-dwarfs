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
print(npes)
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
from gen_halo_exchange import *

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
dynamics.Av = dynamics.Av.at[:,:].set(1.e-3)
stress_surf = stress_surf.at[:,:].set(1.e-3)
C_d=1.e-3
solverinfo = SolverInfo(partit.myDim_nod2D, partit.eDim_nod2D)
rr, zz, pp, App = ssh_solve_preconditioner(solverinfo, partit, mesh)


dynamics.d_eta = jnp.zeros(partit.myDim_nod2D + partit.eDim_nod2D)
print(partit.mype, jnp.sum(mesh.coriolis))

for TIMESTEP in range(2):
    t0 = time.time()
    dynamics.U_rhs, dynamics.V_rhs, dynamics.U_rhsAB, dynamics.V_rhsAB = compute_vel_rhs_opt(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.U_rhsAB, dynamics.V_rhsAB, dynamics.eta_n, dynamics.AB_order, mesh.elem_area, mesh.gradient_sca, mesh.coriolis, mesh.ulevels, mesh.nlevels, mesh.elem2D, partit.myDim_elem2D, g, dt)
    t1 = time.time()

    dynamics.u_c, dynamics.v_c = visc_filt_bilapl_first(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.u_c, dynamics.v_c,
                                        mesh.ulevels, mesh.nlevels, mesh.elem_area, mesh.edge_tri,
                                        dynamics.visc_gamma0, dynamics.visc_gamma1, dynamics.visc_gamma2,
                                        dt, partit.myDim_elem2D, partit.eDim_elem2D, partit.myDim_edge2D, partit.eDim_edge2D)
    t2 = time.time()
    dynamics.u_c = exchange_elem3D(dynamics.u_c, partit)  # needs to be taken out of jax
    dynamics.v_c = exchange_elem3D(dynamics.v_c, partit)  # needs to be taken out of jax
    t3 = time.time()
    # Call the second function (handles external function calls and completes computation)
    dynamics.U_rhs, dynamics.V_rhs, U_c, V_c = visc_filt_bilapl_second(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.u_c,
                                                 dynamics.v_c, mesh.ulevels, mesh.nlevels, mesh.elem_area, mesh.edge_tri,
                                                 partit.myDim_edge2D, partit.eDim_edge2D)
    t4 = time.time()


    dynamics.U_rhs, dynamics.V_rhs = impl_vert_visc_ale_opt(
    dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.w_i,
    stress_surf, dynamics.Av, mesh.elem_area, mesh.elem2D, mesh.ulevels, mesh.nlevels,
    mesh.zbar_e_bot, mesh.helem, C_d, partit.myDim_elem2D, dt)
    t5 = time.time()
 
    dynamics.ssh_rhs = compute_ssh_rhs_ale(
    dynamics.u, dynamics.v,  # U and V velocities
    dynamics.U_rhs, dynamics.V_rhs,  # RHS terms
    dynamics.ssh_rhs, dynamics.ssh_rhs_old,  # SSH terms
    dynamics.water_flux, alpha,  # Water flux and alpha parameter
    mesh.edges, mesh.edge_tri, mesh.edge_cross_dxdy,  # Mesh geometry
    mesh.ulevels, mesh.nlevels, mesh.helem, mesh.areasvol,  # Level and area info
    partit.myDim_nod2D, partit.myDim_edge2D,  # Partition info
    which_ALE  # ALE configuration
    )
    t6 = time.time()
    dynamics.ssh_rhs = exchange_nod2D(dynamics.ssh_rhs, partit)
    t7 = time.time()
    dynamics.d_eta = ssh_solve_cg(dynamics.ssh_rhs, dynamics.d_eta, solverinfo, mesh, partit)
    t8 = time.time()
    # Update velocity field
    dynamics.u, dynamics.v = update_vel(dynamics.u, dynamics.v, dynamics.U_rhs, dynamics.V_rhs, dynamics.d_eta, mesh.elem2D, mesh.gradient_sca, mesh.ulevels, mesh.nlevels, g, theta, dt, partit.myDim_elem2D)
    t9 = time.time()
    # Exchange updated velocities between elements
    dynamics.u = exchange_elem3D(dynamics.u, partit)
    dynamics.v = exchange_elem3D(dynamics.v, partit)
    t10 = time.time()
    # Update hbar using ALE formulation
    mesh.hbar_old, mesh.hbar, dynamics.ssh_rhs_old = compute_hbar_ale(
    dynamics.u, dynamics.v, dynamics.water_flux, mesh.helem,
    mesh.edges, mesh.edge_tri, mesh.edge_cross_dxdy,
    mesh.elem2D, mesh.ulevels, mesh.ulevels_nod2D, mesh.nlevels,
    mesh.area, mesh.hbar_old, mesh.hbar,
    partit.myDim_nod2D, partit.eDim_nod2D, partit.myDim_edge2D,
    partit.myDim_elem2D, dt, dynamics.ssh_rhs_old)
    t11 = time.time()   
    dynamics.ssh_rhs_old=exchange_nod2D(dynamics.ssh_rhs_old, partit)
    mesh.hbar=exchange_nod2D(mesh.hbar, partit)
    t12 = time.time()
    mesh.dhe = compute_dhe_ale(mesh.dhe, mesh.hbar, mesh.hbar_old, partit.myDim_elem2D, mesh.elem2D, mesh.ulevels)
    t13 = time.time()
    # mesh.dhe is allocated only with myDim_elem2D. No exchange needed?
    dynamics.eta_n=alpha*mesh.hbar+(1.0-alpha)*mesh.hbar_old
    t14 = time.time()
    if partit.mype == 0:
        print(f"\nTimestep {TIMESTEP} timing measurements (seconds):")
        print(f"t1-t0: {t1-t0:.6f}")
        print(f"t2-t1: {t2-t1:.6f}")
        print(f"t3-t2: {t3-t2:.6f}")
        print(f"t4-t3: {t4-t3:.6f}")
        print(f"t4-t1: {t4-t1:.6f}")
        print(f"t5-t4: {t5-t4:.6f}")
        print(f"t6-t5: {t6-t5:.6f}")
        print(f"t7-t6: {t7-t6:.6f}")
        print(f"t8-t7: {t8-t7:.6f}")
        print(f"t9-t8: {t9-t8:.6f}")
        print(f"t10-t9: {t10-t9:.6f}")
        print(f"t11-t10: {t11-t10:.6f}")
        print(f"t12-t11: {t12-t11:.6f}")
        print(f"t13-t12: {t13-t12:.6f}")
        print(f"t14-t13: {t14-t13:.6f}")
        print(f"t14-t0: {t14-t0:.6f}")
print("hbar/dhe/ssh_rhs_old", partit.mype, jnp.sum(mesh.hbar), jnp.sum(mesh.dhe), jnp.sum(dynamics.eta_n))        
partit.MPI_COMM_FESOM.Barrier()
MPI.Finalize()
