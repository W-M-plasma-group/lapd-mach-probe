import numpy as np
import astropy.units as u


def radial_profile(diagnostic, steady_state_start, steady_state_end):

    # Develop radial profile; can use in ex. neutrals.py where needed
    linear_diagnostic_profile = linear_profile(diagnostic, steady_state_start, steady_state_end)
    pass


def linear_profile(diagnostic, steady_state_start, steady_state_end):

    if validate_dimensions(diagnostic.sizes):
        time = diagnostic.coords['time'] * (1. * u.Unit(diagnostic.coords['time'].attrs['units'])).to(u.s).value
        diagnostic = diagnostic.squeeze().where(
            np.logical_and(time >= steady_state_start.to(u.s).value, time <= steady_state_end.to(u.s).value), drop=True
        )
        diagnostic = diagnostic.mean(dim='time', keep_attrs=True)
        return diagnostic


def validate_dimensions(da_sizes):
    # check if diagnostic is 1D; needed for radial profile
    if da_sizes['x'] == da_sizes['y'] == 1:
        raise ValueError("Diagnostic data has no spatial dimension. One-dimensional data needed for linear profiles.")
    elif da_sizes['x'] > 1 and da_sizes['y'] > 1:
        print("Linear profiles not defined for two-dimensional (areal) data.")
        return False
    else:
        return True


def get_spatial_dimensions(data_xarray):

    spatial_dimensions = ('x', 'y')
    return tuple(dimension for dimension in spatial_dimensions if data_xarray.sizes[dimension] >= 1)
