import numpy as np
import astropy.units as u
import xarray as xr


def radial_profile(diagnostic, steady_state):

    # Develop radial profile; can use in ex. neutrals.py where needed
    linear_diagnostic_profile = linear_profile(diagnostic, steady_state)
    pass


def between(x, low, high):
    return (x - low) * (x - high) <= 0


def linear_profile(diagnostic: xr.DataArray, steady_state_times, mean=True):

    if validate_dataset_dims(diagnostic.sizes):
        time = diagnostic.coords['time'] * u.Unit(diagnostic.coords['time'].attrs['units'])
        steady_state_da = diagnostic.squeeze().where(between(time, *steady_state_times), drop=True)

        """
        import matplotlib.pyplot as plt  # TODO REMOVE - debug
        if "port" in steady_state_da.dims:
            
            # for probe_da in diagnostic:
            #     # print(probe_da.sizes)
            #     probe_da.squeeze().min(dim='time').plot()
            #     probe_da.squeeze().max(dim='time').plot()
            #     plt.show()
        else:
            # diagnostic.squeeze().min(dim='time').plot()
            # diagnostic.squeeze().max(dim='time').plot()
            # plt.rcParams['figure.figsize'] = (15, 5)
            steady_state_da.squeeze().isel(x=70).rolling(time=100).mean().plot()
            steady_state_da.isel(x=68).rolling(time=100).mean().plot()
            plt.show()
            steady_state_da.plot.contourf(x='time', robust=True)
            plt.show()
        """
        return steady_state_da.mean(dim='time', keep_attrs=True) if mean else steady_state_da


def validate_dataset_dims(diagnostics_dataset_sizes):

    if not np.isin(['x', 'y'], diagnostics_dataset_sizes).any():
        raise ValueError("Dataset does not have x or y dimension.")
    if 'y' not in diagnostics_dataset_sizes or diagnostics_dataset_sizes['y'] == 1:
        linear_dimension = 'x'
    elif 'x' not in diagnostics_dataset_sizes or diagnostics_dataset_sizes['x'] == 1:
        linear_dimension = 'y'
    else:
        raise ValueError("x and y dimensions have lengths " + str(diagnostics_dataset_sizes[:2]) +
                         " both greater than 1. A linear plot cannot be made. Areal plots are not yet supported.")
    if diagnostics_dataset_sizes['time'] == 1:
        raise ValueError("Single-time profiles are not supported")

    return linear_dimension


"""
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
# """
