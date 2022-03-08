import matplotlib.pyplot as plt
from setup import *
from getIsat import *
from velocity import *
from radial import *


# Global parameters
sample_sec = (100 / 16 * 10e6) ** -1 * u.s
steady_state_start_time = 2 * u.ms
steady_state_end_time = 5 * u.ms

# File paths
hdf5_filename = "/Users/leo/Plasma code/HDF5/8-3500A.hdf5"
diagnostic_open_filename = "/Users/leo/Plasma code/lapd-plasma-analysis/netcdf/9-4000A.nc"
diagnostic_save_filename = diagnostic_open_filename
isat_save_filename = "isat_dataarray.nc"
isat_open_filename = isat_save_filename
mach_save_filename = "mach_dataset.nc"
mach_open_filename = mach_save_filename

# User settings
"""Set the use_existing_diagnostic variable to True to open an existing diagnostic dataset,
       or False to create a new one."""
use_existing_diagnostic = True  # False not yet supported
"""Set the use_existing_isat variable to True to open an existing isat DataArray, or False to create a new one."""
use_existing_isat = True
"""Set the save_isat variable to True to save an isat dataset if a new one is created."""
save_isat = False
"""Set the use_existing_isat variable to True to open an existing Mach dataset, or False to create a new one."""
use_existing_mach = True
"""Set the save_isat variable to True to save a Mach dataset if a new one is created."""
save_mach = True
# End user settings


if __name__ == '__main__':
    # TODO raise issue of uTorr/ other misc units not working with SI prefixes
    # TODO still have to create pressure data! (In isweep-vsweep code)

    uTorr = u.def_unit("uTorr", 1e-6 * u.Torr)
    lapd_plot_units = (uTorr, u.gauss, u.kA)
    lapd_parameters = setup_lapd(hdf5_filename, lapd_plot_units)

    diagnostic_dataset = xr.open_dataset(diagnostic_open_filename)
    # Make position-linear electron temperature dataset from LAPD diagnostic dataset
    linear_electron_temperature = linear_profile(diagnostic_dataset['T_e'], steady_state_start_time,
                                                 steady_state_end_time)

    # (Clean up)
    # Check if mach dataset file exists
    generate_mach = not use_existing_mach
    if use_existing_mach:
        try:
            mach_test = xr.open_dataset(mach_open_filename)
        except FileNotFoundError:
            print("** Mach data .nc file not found. Calculating Mach dataset...")
            generate_mach = True
    # Check if isat dataset file exists
    generate_isat = not use_existing_isat
    if generate_mach and use_existing_isat:
        try:
            isat_test = xr.open_dataarray(isat_open_filename)
        except FileNotFoundError:
            print("** Isat data .nc file not found. Calculating Isat data...")
            generate_isat = True

    # Generate a new mach and/or isat dataset file if needed
    if not generate_mach:
        print("Opening Mach data file...")
        mach_diagnostics = xr.open_dataset(mach_open_filename)
    else:
        if not generate_isat:
            print("Opening isat data file...")
            isat = xr.open_dataarray(isat_open_filename)
        else:
            isat = get_isat(hdf5_filename, sample_sec)
            if save_isat:
                isat.to_netcdf(isat_save_filename)
        # mach = get_velocity_profiles(hdf5_filename, sample_sec)
        mach_diagnostics = get_velocity_profiles(isat, linear_electron_temperature, lapd_parameters)
        if save_mach:
            mach_diagnostics.to_netcdf(mach_save_filename)

    # isat[0].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
    # plt.show()

    print("Experimental parameters at LAPD:", {parameter: str(value) for parameter, value in lapd_parameters.items()})
    for variable in mach_diagnostics:
        mach_diagnostics[variable].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
        plt.title(variable)
        plt.show()
        linear_profile(mach_diagnostics[variable].mean(dim='shot', keep_attrs=True),
                       steady_state_start_time, steady_state_end_time).squeeze().plot(x='x')
        plt.title(variable)
        plt.show()

    # plt.title(str([parameter + " = " + str(value) for parameter, value in lapd_parameters.items()]), size='medium')
    # Can use numpy to round experimental parameters and add to all plots now that in correct units?
