import jax.numpy as jnp
from jax import vmap

# Global variable for the transformation matrix
r2g_matrix = jnp.zeros((3, 3))

def set_mesh_transform_matrix(alphaEuler, betaEuler, gammaEuler):
    """
    Set the rotation matrix from Euler angles.
    """
    al = alphaEuler
    be = betaEuler
    ga = gammaEuler

    # Rotation matrix
    global r2g_matrix
    r2g_matrix = jnp.array([
        [jnp.cos(ga)*jnp.cos(al)-jnp.sin(ga)*jnp.cos(be)*jnp.sin(al), jnp.cos(ga)*jnp.sin(al)+jnp.sin(ga)*jnp.cos(be)*jnp.cos(al), jnp.sin(ga)*jnp.sin(be)],
        [-jnp.sin(ga)*jnp.cos(al)-jnp.cos(ga)*jnp.cos(be)*jnp.sin(al), -jnp.sin(ga)*jnp.sin(al)+jnp.cos(ga)*jnp.cos(be)*jnp.cos(al), jnp.cos(ga)*jnp.sin(be)],
        [jnp.sin(be)*jnp.sin(al), -jnp.sin(be)*jnp.cos(al), jnp.cos(be)]
    ])

def r2g(rlon, rlat):
    """
    Transform from rotated coordinates to geographical coordinates.
    """
    xr = jnp.cos(rlat) * jnp.cos(rlon)
    yr = jnp.cos(rlat) * jnp.sin(rlon)
    zr = jnp.sin(rlat)

    xg = r2g_matrix[0, 0] * xr + r2g_matrix[1, 0] * yr + r2g_matrix[2, 0] * zr
    yg = r2g_matrix[0, 1] * xr + r2g_matrix[1, 1] * yr + r2g_matrix[2, 1] * zr
    zg = r2g_matrix[0, 2] * xr + r2g_matrix[1, 2] * yr + r2g_matrix[2, 2] * zr

    glat = jnp.arcsin(zg)
    glon = jnp.where((yg == 0) & (xg == 0), 0.0, jnp.arctan2(yg, xg))
    
    return glon, glat

def g2r(glon, glat):
    """
    Transform from geographical coordinates to rotated coordinates.
    """
    xg = jnp.cos(glat) * jnp.cos(glon)
    yg = jnp.cos(glat) * jnp.sin(glon)
    zg = jnp.sin(glat)

    xr = r2g_matrix[0, 0] * xg + r2g_matrix[0, 1] * yg + r2g_matrix[0, 2] * zg
    yr = r2g_matrix[1, 0] * xg + r2g_matrix[1, 1] * yg + r2g_matrix[1, 2] * zg
    zr = r2g_matrix[2, 0] * xg + r2g_matrix[2, 1] * yg + r2g_matrix[2, 2] * zg

    rlat = jnp.arcsin(zr)
    rlon = jnp.where((yr == 0) & (xr == 0), 0.0, jnp.arctan2(yr, xr))

    return rlon, rlat

def vector_g2r(tlon, tlat, lon, lat, flag_coord):
    """
    Transform a 2D vector from geographical to rotated coordinates.
    """
    if flag_coord == 1:
        glon = lon
        glat = lat
        rlon, rlat = g2r(glon, glat)
    else:
        rlon = lon
        rlat = lat
        glon, glat = r2g(rlon, rlat)
    
    # Vector in Cartesian geographical coordinates
    txg = -tlat * jnp.sin(glat) * jnp.cos(glon) - tlon * jnp.sin(glon)
    tyg = -tlat * jnp.sin(glat) * jnp.sin(glon) + tlon * jnp.cos(glon)
    tzg = tlat * jnp.cos(glat)

    # Vector in Cartesian rotated coordinates
    txr = r2g_matrix[0, 0] * txg + r2g_matrix[0, 1] * tyg + r2g_matrix[0, 2] * tzg
    tyr = r2g_matrix[1, 0] * txg + r2g_matrix[1, 1] * tyg + r2g_matrix[1, 2] * tzg
    tzr = r2g_matrix[2, 0] * txg + r2g_matrix[2, 1] * tyg + r2g_matrix[2, 2] * tzg

    # Vector in rotated coordinates
    tlat_rot = -jnp.sin(rlat) * jnp.cos(rlon) * txr - jnp.sin(rlat) * jnp.sin(rlon) * tyr + jnp.cos(rlat) * tzr
    tlon_rot = -jnp.sin(rlon) * txr + jnp.cos(rlon) * tyr

    return tlon_rot, tlat_rot

def vector_r2g(tlon, tlat, lon, lat, flag_coord):
    """
    Transform a 2D vector from rotated to geographical coordinates.
    """
    if flag_coord == 1:
        glon = lon
        glat = lat
        rlon, rlat = g2r(glon, glat)
    else:
        rlon = lon
        rlat = lat
        glon, glat = r2g(rlon, rlat)
    
    # Vector in Cartesian rotated coordinates
    txg = -tlat * jnp.sin(rlat) * jnp.cos(rlon) - tlon * jnp.sin(rlon)
    tyg = -tlat * jnp.sin(rlat) * jnp.sin(rlon) + tlon * jnp.cos(rlon)
    tzg = tlat * jnp.cos(rlat)

    # Vector in Cartesian geographical coordinates
    txr = r2g_matrix[0, 0] * txg + r2g_matrix[1, 0] * tyg + r2g_matrix[2, 0] * tzg
    tyr = r2g_matrix[0, 1] * txg + r2g_matrix[1, 1] * tyg + r2g_matrix[2, 1] * tzg
    tzr = r2g_matrix[0, 2] * txg + r2g_matrix[1, 2] * tyg + r2g_matrix[2, 2] * tzg

    # Vector in geographical coordinates
    tlat_geo = -jnp.sin(glat) * jnp.cos(glon) * txr - jnp.sin(glat) * jnp.sin(glon) * tyr + jnp.cos(glat) * tzr
    tlon_geo = -jnp.sin(glon) * txr + jnp.cos(glon) * tyr

    return tlon_geo, tlat_geo

def trim_cyclic(b, cyclic_length):
    """
    Trim cyclic value.
    """
    return jnp.where(b > cyclic_length / 2, b - cyclic_length, 
                     jnp.where(b < -cyclic_length / 2, b + cyclic_length, b))

# Example of vectorizing the `r2g` function using vmap:
# Assume `rlon_vec` and `rlat_vec` are arrays of input angles
r2g_vectorized = vmap(r2g, in_axes=(0, 0))  # vectorized version
g2r_vectorized = vmap(g2r, in_axes=(0, 0))  # vectorized version

