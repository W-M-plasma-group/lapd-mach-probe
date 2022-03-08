import numpy as np


def radial_profile(diagnostic, steady_state_start, steady_state_end):

    # Develop radial profile; can use in ex. neutrals.py where needed
    linear_diagnostic_profile = linear_profile(diagnostic, steady_state_start, steady_state_end)
    return


def linear_profile(diagnostic, steady_state_start, steady_state_end):

    # check if diagnostic is 1D; needed for radial profile
    if diagnostic.sizes['x'] == diagnostic.sizes['y'] == 1:
        raise ValueError("Diagnostic data has no spatial dimension. One-dimensional data needed for linear profiles.")
    elif diagnostic.sizes['x'] > 1 and diagnostic.sizes['y'] > 1:
        print("Linear profiles not defined for two-dimensional (areal) data.")
    else:
        time = diagnostic.coords['time']
        return diagnostic.squeeze().where(
            np.logical_and(time >= steady_state_start.value, time <= steady_state_end.value), drop=True
        ).mean(dim='time', keep_attrs=True)


def get_spatial_dimensions(data_xarray):

    spatial_dimensions = ('x', 'y')
    return tuple(dimension for dimension in spatial_dimensions if data_xarray.sizes[dimension] >= 1)
