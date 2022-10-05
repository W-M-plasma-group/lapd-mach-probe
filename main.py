import matplotlib.pyplot as plt
from setup import *
from getMachIsat import *
from velocity import *
from radial import *


# BAPSFLIB parameters
mach_face_bcs = [{2: (3, 1), 5: (3, 3)},
                 {2: (3, 4), 5: (3, 5)}]
mach_receptacles = [3, 4]
mach_face_resistances = [{2: 14.9, 5: 15.0},
                         {2: 14.9, 5: 15.0}]


# Global parameters
steady_state_times = 6 * u.ms, 15 * u.ms  # 2 * u.ms, 5 * u.ms

# File paths
# TODO make custom
hdf5_path = "/Users/leo/lapd-data/March_2022/HDF5/01_line_valves85V_7400A.hdf5"
langmuir_nc_path = "/Users/leo/lapd-data/March_2022/netcdf/01_line_valves85V_7400A.nc"
# hdf5_path = "/Users/leo/lapd-data/March_2022/HDF5/10_line_valves95V_7500A.hdf5"
# langmuir_nc_path = "/Users/leo/lapd-data/March_2022/netcdf/10_line_valves95V_7500A.nc"
mach_save_filename = "/Users/leo/lapd-data/March_2022/mach/mach_da_01.nc"
mach_open_filename = mach_save_filename

# User settings
"""Set the use_existing_diagnostic variable to True to open an existing diagnostic dataset,
       or False to create a new one."""
use_existing_diagnostic = True  # False not yet supported
"""Set the use_existing_mach variable to True to open an existing Mach dataset, or False to create a new one."""
use_existing_mach = True
"""Set the save_mach variable to True to save a Mach dataset if a new one is created."""
save_mach = True
# End user settings


if __name__ == '__main__':
    # TODO raise issue of uTorr/ other misc units not working with SI prefixes

    uTorr = u.def_unit("uTorr", 1e-6 * u.Torr)
    lapd_plot_units = (uTorr, u.gauss, u.kA)
    lapd_parameters = setup_lapd(hdf5_path, lapd_plot_units)

    diagnostic_dataset = xr.open_dataset(langmuir_nc_path)
    # Make position-linear electron temperature dataset from LAPD diagnostic dataset
    linear_electron_temperature = linear_profile(diagnostic_dataset['T_e'], *steady_state_times)

    # (Clean up)
    # Check if mach dataset file exists
    generate_mach = not use_existing_mach
    if use_existing_mach:
        try:
            mach_test = xr.open_dataset(mach_open_filename)
        except FileNotFoundError:
            print("** Mach data .nc file not found. Calculating Mach dataset...")
            generate_mach = True
    # Generate a new mach and/or isat dataset file if needed
    if not generate_mach:
        print("Opening Mach data file...")
        mach_ds = xr.open_dataset(mach_open_filename)
    else:
        mach_isat = get_mach_isat(hdf5_path, mach_face_bcs, mach_receptacles, mach_face_resistances)
        mach_ds = get_velocity_profiles(mach_isat, linear_electron_temperature).assign_attrs(lapd_parameters)
        if save_mach:
            mach_ds.to_netcdf(mach_save_filename)

    # isat[0].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
    # plt.show()

    print("Experimental parameters at LAPD:", {parameter: str(value) for parameter, value in lapd_parameters.items()})
    for probe in range(len(mach_ds.port)):
        for variable in mach_ds.isel(port=probe):
            mach_ds.isel(port=probe)[variable].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf(robust=True)
            title = f"{variable} (Port {mach_ds.port[probe].item()})"
            plt.title(title)
            plt.show()
            linear_profile(mach_ds.isel(port=probe)[variable].mean(dim='shot', keep_attrs=True),
                           *steady_state_times).squeeze().plot(x='x')
            plt.title(title)
            plt.show()

    # plt.title(str([parameter + " = " + str(value) for parameter, value in lapd_parameters.items()]), size='medium')
    # Can use numpy to round experimental parameters and add to all plots now that in correct units?
