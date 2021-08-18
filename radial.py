import xarray as xr


def get_radial_profile(diagnostic, steady_state_start, steady_state_end):

    linear_profile = get_linear_profile(diagnostic, steady_state_start, steady_state_end)
    return


def get_linear_profile(diagnostic, steady_state_start, steady_state_end):

    # check if diagnostic is 1D; needed for radial profile
    if diagnostic.sizes['x'] == diagnostic.sizes['y'] == 1:
        raise ValueError("Diagnostic data has no spatial dimension. One-dimensional data needed for linear profiles.")
    elif diagnostic.sizes['x'] > 1 and diagnostic.sizes['y'] > 1:
        print("Linear profiles not defined for two-dimensional (areal) data.")
    else:
        both = xr.ufuncs.logical_and
        time = diagnostic.coords['time']
        return diagnostic.squeeze().where(
            both(time >= steady_state_start.value, time <= steady_state_end.value), drop=True).mean(dim='time')


def get_spatial_dimensions(data_xarray):

    spatial_dimensions = ('x', 'y')
    return tuple(dimension for dimension in spatial_dimensions if data_xarray.sizes[dimension] >= 1)

