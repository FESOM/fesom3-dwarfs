import jax
import jax.numpy as jnp
from jax import lax
from mpi4py import MPI
#from mpi4jax import send, recv, bcast
from module_rotate_grid import *
from jax import jit
from read_mesh_and_partition import *
from jax import debug

def compute_vel_rhs_opt(U, V, U_rhs, V_rhs, U_rhsAB, V_rhsAB, eta_n, AB_order, elem_area, gradient_sca, coriolis, ulevels, nlevels, elem2D, myDim_elem2D, g, dt):
    ab_coefficients = {
        2: (-0.5, 1.5, 0.0),
        3: (5.0 / 12.0, -16.0 / 12.0, 23.0 / 12.0)
    }

    ab1, ab2, ab3 = ab_coefficients.get(AB_order, (None, None, None))
    if ab1 is None:
        raise ValueError("Unsupported AB scheme. Use 2 or 3.")

    def process_element(elem, state):
        U_rhs, V_rhs, U_rhsAB, V_rhsAB = state
        nzmin = ulevels[elem] - 1  # Convert from Fortran to Python 0-based indexing
        nzmax = nlevels[elem]      # Already adjusted for Python indexing

        def ab_loop(nz, state):
            U_rhs, V_rhs = state
            if AB_order == 2:
                U_rhs = U_rhs.at[nz, elem].set(ab1 * U_rhsAB[nz, elem, 0])
                V_rhs = V_rhs.at[nz, elem].set(ab1 * V_rhsAB[nz, elem, 0])
            elif AB_order == 3:
                U_rhs = U_rhs.at[nz, elem].set(
                    ab1 * U_rhsAB[nz, elem, 1] + ab2 * U_rhsAB[nz, elem, 0])
                V_rhs = V_rhs.at[nz, elem].set(
                    ab1 * V_rhsAB[nz, elem, 1] + ab2 * V_rhsAB[nz, elem, 0])
            return U_rhs, V_rhs

        state = (U_rhs, V_rhs)
        U_rhs, V_rhs = lax.fori_loop(nzmin, nzmax, ab_loop, state)

        elnodes = elem2D[:, elem]
        pre = -g * eta_n[elnodes]
        ff = coriolis[elem] * elem_area[elem]
        Fx = jnp.sum(gradient_sca[:3, elem] * pre)
        Fy = jnp.sum(gradient_sca[3:, elem] * pre)

        def rhs_loop(nz, state):
            U_rhs, V_rhs, U_rhsAB, V_rhsAB = state
            U_rhs = U_rhs.at[nz, elem].add(Fx * elem_area[elem])
            V_rhs = V_rhs.at[nz, elem].add(Fy * elem_area[elem])
            if AB_order == 2:
                U_rhsAB = U_rhsAB.at[nz, elem, 0].set(V[nz, elem] * ff)
                V_rhsAB = V_rhsAB.at[nz, elem, 0].set(-U[nz, elem] * ff)
            elif AB_order == 3:
                U_rhsAB = U_rhsAB.at[nz, elem, 1].set(U_rhsAB[nz, elem, 0])
                V_rhsAB = V_rhsAB.at[nz, elem, 1].set(V_rhsAB[nz, elem, 0])
                U_rhsAB = U_rhsAB.at[nz, elem, 0].set(V[nz, elem] * ff)
                V_rhsAB = V_rhsAB.at[nz, elem, 0].set(-U[nz, elem] * ff)
            return U_rhs, V_rhs, U_rhsAB, V_rhsAB

        U_rhs, V_rhs, U_rhsAB, V_rhsAB = lax.fori_loop(
            nzmin, nzmax, rhs_loop, (U_rhs, V_rhs, U_rhsAB, V_rhsAB)
        )
        return U_rhs, V_rhs, U_rhsAB, V_rhsAB

    state = (U_rhs, V_rhs, U_rhsAB, V_rhsAB)
    state = lax.fori_loop(0, myDim_elem2D, process_element, state)
    U_rhs, V_rhs, U_rhsAB, V_rhsAB = state

    # Update velocity RHS
    ff = ab2 if AB_order == 2 else ab3

    def update_rhs(elem, state):
        U_rhs, V_rhs, U_rhsAB, V_rhsAB = state
        nzmin = ulevels[elem] - 1  # Convert from Fortran to Python 0-based indexing
        nzmax = nlevels[elem]      # Already adjusted for Python indexing

        def update_loop(nz, state):
            U_rhs, V_rhs = state
            U_rhs = U_rhs.at[nz, elem].set(
                dt * (U_rhs[nz, elem] + U_rhsAB[nz, elem, 0] * ff) / elem_area[elem]
            )
            V_rhs = V_rhs.at[nz, elem].set(
                dt * (V_rhs[nz, elem] + V_rhsAB[nz, elem, 0] * ff) / elem_area[elem]
            )
            return U_rhs, V_rhs

        U_rhs, V_rhs = lax.fori_loop(nzmin, nzmax, update_loop, (U_rhs, V_rhs))
        return U_rhs, V_rhs, U_rhsAB, V_rhsAB

    state = lax.fori_loop(0, myDim_elem2D, update_rhs, state)
    U_rhs, V_rhs, U_rhsAB, V_rhsAB = state

    return U_rhs, V_rhs, U_rhsAB, V_rhsAB
compute_vel_rhs_opt_jit = jit(compute_vel_rhs_opt, static_argnums=(7,))

def visc_filt_bilapl(u, v, U_rhs, V_rhs, U_c, V_c, ulevels, nlevels, elem_area, edge_tri,
                     visc_gamma0, visc_gamma1, visc_gamma2, dt, myDim_elem2D, eDim_elem2D, myDim_edge2D, eDim_edge2D):
    # Step 1: Reset U_c and V_c
    U_c = U_c.at[:, :].set(0.0)
    V_c = V_c.at[:, :].set(0.0)

    # Maximum depth
    max_depth = u.shape[0]

    # Step 2: Sum velocity differences over edges
    def process_edge(ed, state):
        U_c, V_c = state
        el = edge_tri[:, ed]
        nzmin = jnp.max(ulevels[el]) - 1
        nzmax = jnp.min(nlevels[el])

        # Fixed-size buffer
        update_u = jnp.zeros((max_depth,))
        update_v = jnp.zeros((max_depth,))

        def compute_updates(nz, updates):
            update_u, update_v = updates
            du = u[nz, el[0]] - u[nz, el[1]]
            dv = v[nz, el[0]] - v[nz, el[1]]
            updates = (
                update_u.at[nz].set(du),
                update_v.at[nz].set(dv),
            )
            return updates

        update_u, update_v = lax.fori_loop(nzmin, nzmax, compute_updates, (update_u, update_v))

        # Mask to apply only to the valid range
        mask = (jnp.arange(max_depth) >= nzmin) & (jnp.arange(max_depth) < nzmax)

        # Apply updates with masking
        U_c = U_c.at[:, el[0]].add(jnp.where(mask, -update_u, 0))
        V_c = V_c.at[:, el[0]].add(jnp.where(mask, -update_v, 0))
        U_c = U_c.at[:, el[1]].add(jnp.where(mask, update_u, 0))
        V_c = V_c.at[:, el[1]].add(jnp.where(mask, update_v, 0))

        return U_c, V_c

    U_c, V_c = lax.fori_loop(0, myDim_edge2D + eDim_edge2D, process_edge, (U_c, V_c))

    # Step 3: Compute viscosity on elements
    def process_element(elem, state):
        U_c, V_c = state
        len_elem = jnp.sqrt(elem_area[elem])
        nzmin = ulevels[elem]-1
        nzmax = nlevels[elem]

        def update_viscosity(nz, state):
            U_c, V_c = state
            u1 = U_c[nz, elem]**2 + V_c[nz, elem]**2
            vi = jnp.max(jnp.array([
                visc_gamma0,
                visc_gamma1 * jnp.sqrt(u1),
                visc_gamma2 * u1
            ])) * len_elem * dt

            U_c = U_c.at[nz, elem].set(-U_c[nz, elem] * vi)
            V_c = V_c.at[nz, elem].set(-V_c[nz, elem] * vi)
            return U_c, V_c

        U_c, V_c = lax.fori_loop(nzmin, nzmax, update_viscosity, (U_c, V_c))
        return U_c, V_c

    U_c, V_c = lax.fori_loop(0, myDim_elem2D, process_element, (U_c, V_c))

#    U_c = exchange_elem3D(U_c, partit) # needs to be taken out of jax
#    V_c = exchange_elem3D(V_c, partit) # needs to be taken out of jax
    # Step 4: Update U_rhs and V_rhs based on U_c and V_c
    def update_rhs(ed, state):
        U_rhs, V_rhs = state
        el = edge_tri[:, ed]
        nzmin = jnp.max(ulevels[el]) - 1
        nzmax = jnp.min(nlevels[el])
        # Maximum depth
        max_depth = u.shape[0]
        update_u = jnp.zeros((max_depth,))
        update_v = jnp.zeros((max_depth,))

        def compute_rhs_updates(nz, updates):
            update_u, update_v = updates
            du = U_c[nz, el[0]] - U_c[nz, el[1]]
            dv = V_c[nz, el[0]] - V_c[nz, el[1]]
            updates = (
                update_u.at[nz].set(du),
                update_v.at[nz].set(dv),
            )
            return updates

        update_u, update_v = lax.fori_loop(nzmin, nzmax, compute_rhs_updates, (update_u, update_v))

        # Mask to apply only to the valid range
        mask = (jnp.arange(max_depth) >= nzmin) & (jnp.arange(max_depth) < nzmax)

        # Apply updates with masking
        U_rhs = U_rhs.at[:, el[0]].add(jnp.where(mask, -update_u / elem_area[el[0]], 0))
        V_rhs = V_rhs.at[:, el[0]].add(jnp.where(mask, -update_v / elem_area[el[0]], 0))
        U_rhs = U_rhs.at[:, el[1]].add(jnp.where(mask, update_u / elem_area[el[1]], 0))
        V_rhs = V_rhs.at[:, el[1]].add(jnp.where(mask, update_v / elem_area[el[1]], 0))

        return U_rhs, V_rhs

    U_rhs, V_rhs = lax.fori_loop(0, myDim_edge2D + eDim_edge2D, update_rhs, (U_rhs, V_rhs))

    return U_rhs, V_rhs, U_c, V_c
visc_filt_bilapl_jit = jax.jit(visc_filt_bilapl)

def visc_filt_bilapl_first(u, v, U_rhs, V_rhs, U_c, V_c, ulevels, nlevels, elem_area, edge_tri,
                           visc_gamma0, visc_gamma1, visc_gamma2, dt, myDim_elem2D, eDim_elem2D, myDim_edge2D, eDim_edge2D):
    # Step 1: Reset U_c and V_c
    U_c = U_c.at[:, :].set(0.0)
    V_c = V_c.at[:, :].set(0.0)

    # Maximum depth
    max_depth = u.shape[0]

    # Step 2: Sum velocity differences over edges
    def process_edge(ed, state):
        U_c, V_c = state
        el = edge_tri[:, ed]
        el = jnp.where(el[1] < 0, el.at[1].set(el[0]), el)
        nzmin = jnp.max(ulevels[el]) - 1
        nzmax = jnp.min(nlevels[el])

        # Fixed-size buffer
        update_u = jnp.zeros((max_depth,))
        update_v = jnp.zeros((max_depth,))

        def compute_updates(nz, updates):
            update_u, update_v = updates
            du = u[nz, el[0]] - u[nz, el[1]]
            dv = v[nz, el[0]] - v[nz, el[1]]
            updates = (
                update_u.at[nz].set(du),
                update_v.at[nz].set(dv),
            )
            return updates

        update_u, update_v = lax.fori_loop(nzmin, nzmax, compute_updates, (update_u, update_v))

        # Mask to apply only to the valid range
        mask = (jnp.arange(max_depth) >= nzmin) & (jnp.arange(max_depth) < nzmax)

        # Apply updates with masking
        U_c = U_c.at[:, el[0]].add(jnp.where(mask, -update_u, 0))
        V_c = V_c.at[:, el[0]].add(jnp.where(mask, -update_v, 0))
        U_c = U_c.at[:, el[1]].add(jnp.where(mask, update_u, 0))
        V_c = V_c.at[:, el[1]].add(jnp.where(mask, update_v, 0))

        return U_c, V_c

    U_c, V_c = lax.fori_loop(0, myDim_edge2D + eDim_edge2D, process_edge, (U_c, V_c))

    # Step 3: Compute viscosity on elements
    def process_element(elem, state):
        U_c, V_c = state
        len_elem = jnp.sqrt(elem_area[elem])
        nzmin = ulevels[elem]-1
        nzmax = nlevels[elem]

        def update_viscosity(nz, state):
            U_c, V_c = state
            u1 = U_c[nz, elem]**2 + V_c[nz, elem]**2
            vi = jnp.max(jnp.array([
                visc_gamma0,
                visc_gamma1 * jnp.sqrt(u1),
                visc_gamma2 * u1
            ])) * len_elem * dt

            U_c = U_c.at[nz, elem].set(-U_c[nz, elem] * vi)
            V_c = V_c.at[nz, elem].set(-V_c[nz, elem] * vi)
            return U_c, V_c

        U_c, V_c = lax.fori_loop(nzmin, nzmax, update_viscosity, (U_c, V_c))
        return U_c, V_c

    U_c, V_c = lax.fori_loop(0, myDim_elem2D, process_element, (U_c, V_c))
    return U_c, V_c
visc_filt_bilapl_first_jit = jax.jit(visc_filt_bilapl_first)

def visc_filt_bilapl_second(u, v, U_rhs, V_rhs, U_c, V_c, ulevels, nlevels, elem_area, edge_tri,
                            myDim_edge2D, eDim_edge2D):
    def update_rhs(ed, state):
        U_rhs, V_rhs = state
        el = edge_tri[:, ed]
        el = jnp.where(el[1] < 0, el.at[1].set(el[0]), el)
        nzmin = jnp.max(ulevels[el]) - 1
        nzmax = jnp.min(nlevels[el])
        # Maximum depth
        max_depth = u.shape[0]
        update_u = jnp.zeros((max_depth,))
        update_v = jnp.zeros((max_depth,))

        def compute_rhs_updates(nz, updates):
            update_u, update_v = updates
            du = U_c[nz, el[0]] - U_c[nz, el[1]]
            dv = V_c[nz, el[0]] - V_c[nz, el[1]]
            updates = (
                update_u.at[nz].set(du),
                update_v.at[nz].set(dv),
            )
            return updates

        update_u, update_v = lax.fori_loop(nzmin, nzmax, compute_rhs_updates, (update_u, update_v))

        # Mask to apply only to the valid range
        mask = (jnp.arange(max_depth) >= nzmin) & (jnp.arange(max_depth) < nzmax)

        # Apply updates with masking
        U_rhs = U_rhs.at[:, el[0]].add(jnp.where(mask, -update_u / elem_area[el[0]], 0))
        V_rhs = V_rhs.at[:, el[0]].add(jnp.where(mask, -update_v / elem_area[el[0]], 0))
        U_rhs = U_rhs.at[:, el[1]].add(jnp.where(mask, update_u / elem_area[el[1]], 0))
        V_rhs = V_rhs.at[:, el[1]].add(jnp.where(mask, update_v / elem_area[el[1]], 0))

        return U_rhs, V_rhs

    U_rhs, V_rhs = lax.fori_loop(0, myDim_edge2D + eDim_edge2D, update_rhs, (U_rhs, V_rhs))

    return U_rhs, V_rhs, U_c, V_c
visc_filt_bilapl_second_jit = jax.jit(visc_filt_bilapl_second)


def impl_vert_visc_ale_opt_jit(U, V, U_rhs, V_rhs, Wvel_i, stress_surf, Av, elem_area, elem2D, ulevels, nlevels,
                               zbar_e_bot,
                               helem, C_d, myDim_elem2D, dt):
    max_nz = U.shape[0]  # Calculate max_nz dynamically based on the input shape
    eps = 1e-10  # Small value to prevent division by zero
    density_0 = 1030.0  # Reference density

    def process_element(elem, state):
        U_rhs, V_rhs = state
        elnodes = elem2D[:, elem]
        nzmin = ulevels[elem] - 1  # Convert from Fortran to Python 0-based indexing
        nzmax = nlevels[elem]      # Already adjusted for Python indexing
        # Initialize arrays with zeros explicitly
        zbar_n = jnp.zeros(max_nz + 1, dtype=jnp.float64)  # For levels: indices 0-14 (15 levels)
        Z_n = jnp.zeros(max_nz, dtype=jnp.float64)         # For layers: indices 0-13 (14 layers)
        a = jnp.zeros(max_nz, dtype=jnp.float64)           # For layers
        b = jnp.zeros(max_nz, dtype=jnp.float64)           # For layers
        c = jnp.zeros(max_nz, dtype=jnp.float64)           # For layers
        ur = jnp.zeros(max_nz, dtype=jnp.float64)          # For layers
        vr = jnp.zeros(max_nz, dtype=jnp.float64)          # For layers

        # Initialize vertical grid from top to bottom
        zbar_n = zbar_n.at[nzmin].set(0.0)  # Set surface level to 0
        
        def forward_initialize_zbar_Z(i, zbar_state):
            zbar_n, Z_n = zbar_state
            nz = nzmin + i  # Current level index
            
            # Update zbar_n for the next level
            is_valid_level = (nz + 1) <= nzmax
            zbar_n = lax.cond(is_valid_level,
                             lambda x: x.at[nz + 1].set(x[nz] - helem[nz, elem]),
                             lambda x: x,
                             zbar_n)
            
            # Update Z_n (midpoint) for the current layer
            is_valid_layer = nz < nzmax
            Z_n = lax.cond(is_valid_layer,
                          lambda x: x.at[nz].set((zbar_n[nz] + zbar_n[nz + 1]) / 2.0),
                          lambda x: x,
                          Z_n)
            
            return zbar_n, Z_n

        # Initialize vertical grid
        zbar_n, Z_n = lax.fori_loop(0, nzmax - nzmin + 1, forward_initialize_zbar_Z, (zbar_n, Z_n))
#        debug.print("zbar_n: {}", zbar_n)
#        debug.print("Z_n: {}", Z_n)
        # Compute coefficients
        def compute_coefficients(nz, coeff_state):
            a, b, c = coeff_state
            dz = zbar_n[nz] - zbar_n[nz + 1]  # Using levels
            zinv = dt / dz
            dz_up = Z_n[nz - 1] - Z_n[nz]     # Using layers
            dz_down = Z_n[nz] - Z_n[nz + 1]   # Using layers

            a = a.at[nz].set(-Av[nz, elem] / dz_up * zinv)
            c = c.at[nz].set(-Av[nz + 1, elem] / dz_down * zinv)
            b = b.at[nz].set(-a[nz] - c[nz] + 1.0)

            # Vertical advection updates
            wu = jnp.sum(lax.dynamic_slice(Wvel_i, (nz, elnodes[0]), (1, 3))) / 3.0
            wd = jnp.sum(lax.dynamic_slice(Wvel_i, (nz + 1, elnodes[0]), (1, 3))) / 3.0
            a = a.at[nz].add(jnp.minimum(0.0, wu) * zinv)
            b = b.at[nz].add(jnp.maximum(0.0, wu) * zinv - jnp.minimum(0.0, wd) * zinv)
            c = c.at[nz].add(-jnp.maximum(0.0, wd) * zinv)
            return a, b, c

        # Compute coefficients for interior points
        def interior_loop(i, state):
            a, b, c = state
            nz = nzmin + 1 + i
            is_valid = (nz < nzmax - 1) & (nz >= nzmin + 1)  # For layers
            state = lax.cond(is_valid,
                           lambda s: compute_coefficients(nz, s),
                           lambda s: s,
                           state)
            return state

        a, b, c = lax.fori_loop(0, max_nz - 2, interior_loop, (a, b, c))

        # Last row - using layers
        dz = zbar_n[nzmax - 1] - zbar_n[nzmax]  # Using levels
        zinv = dt / dz
        dz_up = Z_n[nzmax - 2] - Z_n[nzmax - 1]  # Using layers
        a = a.at[nzmax - 1].set(-Av[nzmax - 1, elem] / dz_up * zinv)
        b = b.at[nzmax - 1].set(-a[nzmax - 1] + 1.0)
        c = c.at[nzmax - 1].set(0.0)

        wu = jnp.sum(lax.dynamic_slice(Wvel_i, (nzmax - 1, elnodes[0]), (1, 3))) / 3.0
        a = a.at[nzmax - 1].add(jnp.minimum(0.0, wu) * zinv)
        b = b.at[nzmax - 1].add(jnp.maximum(0.0, wu) * zinv)

        # First row
        dz = zbar_n[nzmin] - zbar_n[nzmin + 1]  # Using levels
        zinv = dt / dz
        dz_down = Z_n[nzmin] - Z_n[nzmin + 1]   # Using layers
        c = c.at[nzmin].set(-Av[nzmin + 1, elem] / dz_down * zinv)
        a = a.at[nzmin].set(0.0)
        b = b.at[nzmin].set(-c[nzmin] + 1.0)

        wd = jnp.sum(lax.dynamic_slice(Wvel_i, (nzmin + 1, elnodes[0]), (1, 3))) / 3.0
        b = b.at[nzmin].add(jnp.maximum(0.0, wd) * zinv)
        c = c.at[nzmin].add(-jnp.maximum(0.0, wd) * zinv)

        # Update RHS for first row
        ur = ur.at[nzmin].set(U_rhs[nzmin, elem] - (b[nzmin] - 1.0) * U[nzmin, elem] - c[nzmin] * U[nzmin + 1, elem])
        vr = vr.at[nzmin].set(V_rhs[nzmin, elem] - (b[nzmin] - 1.0) * V[nzmin, elem] - c[nzmin] * V[nzmin + 1, elem])

        # Update RHS - using layers
        def update_rhs_range(i, state):
            ur, vr = state
            nz = nzmin + i
            is_valid = (nz < nzmax) & (nz >= nzmin)  # For layers
            ur = lax.cond(is_valid,
                         lambda x: x.at[nz].set(U_rhs[nz, elem]),
                         lambda x: x,
                         ur)
            vr = lax.cond(is_valid,
                         lambda x: x.at[nz].set(V_rhs[nz, elem]),
                         lambda x: x,
                         vr)
            return ur, vr

        ur, vr = lax.fori_loop(0, max_nz, update_rhs_range, (ur, vr))

        # Add first layer RHS update to match Fortran
        ur = ur.at[nzmin].add(-(b[nzmin]-1.0)*U[nzmin, elem] - c[nzmin]*U[nzmin+1, elem])
        vr = vr.at[nzmin].add(-(b[nzmin]-1.0)*V[nzmin, elem] - c[nzmin]*V[nzmin+1, elem])

        # Add surface forcing with density_0 - using layers
        ur = ur.at[nzmin].add(zinv * stress_surf[0, elem] / density_0)
        vr = vr.at[nzmin].add(zinv * stress_surf[1, elem] / density_0)

        # Add bottom friction - using layers
        zinv = dt / (zbar_n[nzmax - 1] - zbar_n[nzmax])  # Using levels
        bottom_vel = jnp.sqrt(U[nzmax - 1, elem] ** 2 + V[nzmax - 1, elem] ** 2)  # Bottom layer
        friction = -C_d * bottom_vel
        ur = ur.at[nzmax - 1].add(zinv * friction * U[nzmax - 1, elem])  # Bottom layer
        vr = vr.at[nzmax - 1].add(zinv * friction * V[nzmax - 1, elem])  # Bottom layer

        # Update RHS for advective and diffusive contributions
        def update_interior(nz, rhs_state):
            ur, vr = rhs_state
            is_valid = (nz > nzmin) & (nz < nzmax - 1)
            ur = lax.cond(is_valid,
                         lambda x: x.at[nz].add(-a[nz] * U[nz - 1, elem] - (b[nz] - 1.0) * U[nz, elem] - c[nz] * U[nz + 1, elem]),
                         lambda x: x,
                         ur)
            vr = lax.cond(is_valid,
                         lambda x: x.at[nz].add(-a[nz] * V[nz - 1, elem] - (b[nz] - 1.0) * V[nz, elem] - c[nz] * V[nz + 1, elem]),
                         lambda x: x,
                         vr)
            return ur, vr

        ur, vr = lax.fori_loop(nzmin + 1, nzmax - 1, update_interior, (ur, vr))

        # Update bottom layer RHS separately (match Fortran)
        ur = ur.at[nzmax - 1].add(-a[nzmax - 1] * U[nzmax - 2, elem] - (b[nzmax - 1] - 1.0) * U[nzmax - 1, elem])
        vr = vr.at[nzmax - 1].add(-a[nzmax - 1] * V[nzmax - 2, elem] - (b[nzmax - 1] - 1.0) * V[nzmax - 1, elem])

        # Initialize sweep algorithm arrays
        cp = jnp.zeros(max_nz, dtype=jnp.float64)
        up = jnp.zeros(max_nz, dtype=jnp.float64)
        vp = jnp.zeros(max_nz, dtype=jnp.float64)

        # Forward sweep (match Fortran)
        cp = cp.at[nzmin].set(c[nzmin] / b[nzmin])
        up = up.at[nzmin].set(ur[nzmin] / b[nzmin])
        vp = vp.at[nzmin].set(vr[nzmin] / b[nzmin])

        def forward_sweep(i, sweep_state):
            cp, up, vp = sweep_state
            nz = nzmin + 1 + i
            is_valid = (nz < nzmax) & (nz >= nzmin)

            def update_sweep(state):
                cp, up, vp = state
                m = b[nz] - cp[nz - 1] * a[nz]
                cp = cp.at[nz].set(c[nz] / m)
                up = up.at[nz].set((ur[nz] - up[nz - 1] * a[nz]) / m)
                vp = vp.at[nz].set((vr[nz] - vp[nz - 1] * a[nz]) / m)
                return cp, up, vp

            return lax.cond(is_valid,
                          lambda s: update_sweep(s),
                          lambda s: s,
                          sweep_state)

        # Forward sweep from nzmin+1 to nzmax-1
        cp, up, vp = lax.fori_loop(0, nzmax - nzmin - 1, forward_sweep, (cp, up, vp))

        # Back substitution (match Fortran)
        ur = ur.at[nzmax - 1].set(up[nzmax - 1])
        vr = vr.at[nzmax - 1].set(vp[nzmax - 1])

        def backward_substitution(i, back_state):
            ur, vr = back_state
            nz = nzmax - 2 - i
            is_valid = (nz >= nzmin) & (nz < nzmax - 1)
            ur = lax.cond(is_valid,
                         lambda x: x.at[nz].set(up[nz] - cp[nz] * ur[nz + 1]),
                         lambda x: x,
                         ur)
            vr = lax.cond(is_valid,
                         lambda x: x.at[nz].set(vp[nz] - cp[nz] * vr[nz + 1]),
                         lambda x: x,
                         vr)
            return ur, vr

        # Back substitution from nzmax-2 down to nzmin
        ur, vr = lax.fori_loop(0, nzmax - nzmin - 1, backward_substitution, (ur, vr))

        # Update final RHS
        def update_final_rhs(i, rhs_state):
            U_rhs, V_rhs = rhs_state
            nz = nzmin + i
            is_valid = (nz < nzmax) & (nz >= nzmin)
            U_rhs = lax.cond(is_valid,
                           lambda x: x.at[nz, elem].set(ur[nz]),
                           lambda x: x,
                           U_rhs)
            V_rhs = lax.cond(is_valid,
                           lambda x: x.at[nz, elem].set(vr[nz]),
                           lambda x: x,
                           V_rhs)
            return U_rhs, V_rhs

        # Update RHS for all valid levels
        U_rhs, V_rhs = lax.fori_loop(0, nzmax - nzmin, update_final_rhs, (U_rhs, V_rhs))

        return U_rhs, V_rhs

    state = (U_rhs, V_rhs)
    state = lax.fori_loop(0, myDim_elem2D, process_element, state)
    U_rhs, V_rhs = state
    return U_rhs, V_rhs

def compute_ssh_rhs_ale(u, v, u_rhs, v_rhs, ssh_rhs, ssh_rhs_old, water_flux, alpha, edges, edge_tri, edge_cross_dxdy,
                       ulevels, nlevels, helem, areasvol, myDim_nod2D, myDim_edge2D, which_ALE):
    # Initialize ssh_rhs to zero
    ssh_rhs = ssh_rhs.at[:].set(0.0)

    def process_edge(ed, ssh_rhs):
        # Get nodes and elements for this edge
        enodes = edges[:, ed]
        el = edge_tri[:, ed]
        
        # Calculate depth integral for el[0]
        c1 = 0.0
        deltaX1 = edge_cross_dxdy[0, ed]
        deltaY1 = edge_cross_dxdy[1, ed]
        
        nzmin = ulevels[el[0]] - 1  # Convert from Fortran to Python 0-based indexing
        nzmax = nlevels[el[0]] - 1  # Match Fortran's nzmax = nlevels(el(1))-1
        
        def integrate_flux_el1(nz, c1):
            # Compute flux contribution for element 1
            # Note: In Fortran UV(1,:,:) is U and UV(2,:,:) is V
            flux = alpha * ((v[nz, el[0]] + v_rhs[nz, el[0]]) * deltaX1 - 
                          (u[nz, el[0]] + u_rhs[nz, el[0]]) * deltaY1) * helem[nz, el[0]]
            return c1 + flux

        c1 = lax.fori_loop(nzmin, nzmax + 1, integrate_flux_el1, c1)  # +1 to match Fortran inclusive range
        
        # Calculate depth integral for el[1] if it exists
        c2 = 0.0
        def integrate_flux_el2(nz, c2):
            deltaX2 = edge_cross_dxdy[2, ed]
            deltaY2 = edge_cross_dxdy[3, ed]
            # Compute flux contribution for element 2
            flux = -alpha * ((v[nz, el[1]] + v_rhs[nz, el[1]]) * deltaX2 - 
                           (u[nz, el[1]] + u_rhs[nz, el[1]]) * deltaY2) * helem[nz, el[1]]
            return c2 + flux

        # Only compute for el[1] if it exists (>= 0)
        c2 = lax.cond(el[1] >=0,
                     lambda x: lax.fori_loop(ulevels[el[1]] - 1, nlevels[el[1]], integrate_flux_el2, x),
                     lambda x: x,
                     c2)

        # Update ssh_rhs for both nodes
        ssh_rhs = ssh_rhs.at[enodes[0]].add(c1 + c2)
        ssh_rhs = ssh_rhs.at[enodes[1]].add(-(c1 + c2))
        
        return ssh_rhs

    # Process all edges
    ssh_rhs = lax.fori_loop(0, myDim_edge2D, process_edge, ssh_rhs)

    # Handle water flux boundary conditions
    def process_node(n, ssh_rhs):
        nzmin = ulevels[n] - 1  # Convert from Fortran to Python 0-based indexing
        
        # Add water flux and old rhs at surface if not using linfs scheme
        def handle_surface(ssh_rhs):
            ssh_rhs = ssh_rhs.at[n].add(
                -alpha * water_flux[n] * areasvol[nzmin, n] + 
                (1.0 - alpha) * ssh_rhs_old[n]
            )
            return ssh_rhs
        
        # Skip cavity points if which_ALE is 'linfs'
        def handle_linfs(ssh_rhs):
            ssh_rhs = lax.cond(ulevels[n] == 1,
                             lambda x: x.at[n].add((1.0 - alpha) * ssh_rhs_old[n]),
                             lambda x: x,
                             ssh_rhs)
            return ssh_rhs
        
        # Apply appropriate water flux handling based on ALE scheme
        ssh_rhs = lax.cond(which_ALE == 'linfs',
                          lambda x: handle_linfs(x),
                          lambda x: handle_surface(x),
                          ssh_rhs)
        
        return ssh_rhs

    # Process water flux for all nodes
    ssh_rhs = lax.fori_loop(0, myDim_nod2D, process_node, ssh_rhs)
    # MPI exchange to synchronize ssh_rhs across ranks
    # Note: This is handled by the caller in read_mesh_jax.py
    
    return ssh_rhs
compute_ssh_rhs_ale_jit = jax.jit(compute_ssh_rhs_ale, static_argnames=['which_ALE'])
