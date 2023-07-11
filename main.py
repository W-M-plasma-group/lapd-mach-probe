from experimental import *
from getMachIsat import *
from velocity import *
from files import *
from plots import *

# BAPSFLIB parameters
mach_face_bcs = [{2: (3, 1), 5: (3, 3)},
                 {2: (3, 4), 5: (3, 5)}]
mach_receptacles = [3, 4]
mach_face_resistances = [{2: 14.9, 5: 15.0},
                         {2: 14.9, 5: 15.0}]


# Global parameters
core_radius = 26 * u.cm
ion = 'He-4+'

# File paths  # TODO make custom
hdf5_folder = "/Users/leo/lapd-data/March_2022/"
lang_nc_folder = hdf5_folder + "lang_nc/"
mach_nc_folder = hdf5_folder + "mach_nc/"

# User file settings
"""Set the save_mach variable to True to save a Mach dataset if a new one is created."""
save_mach_ds = True
# End user file settings


# TODO combine into one large GitHub project


def port_selector(ds):  # TODO allow multiple modified datasets to be returned
    # port_list = dataset.port  # use if switch to dataset.sel
    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    ds_ports_selected = ds.isel(port=1)  # ds  # - ds.isel(port=1)  # TODO user change for ex. delta-P-parallel
    return f"Port {ds_ports_selected.port.item()}", ds_ports_selected.assign_attrs(manual_attrs)
    # use the "exec" function to prompt user to input desired transformation? Or ask for a linear transformation
    # Add a string attribute to the dataset to describe which port(s) comes from


if __name__ == "__main__":

    mach_nc_folder = ensure_directory(mach_nc_folder)  # Create and check folder to save NetCDF files if not yet existing

    print("The following Mach diagnostic NetCDF files were found in the Mach NetCDF folder (specified in main.py): ")
    mach_nc_paths = sorted(search_folder(mach_nc_folder, 'nc', limit=26))
    mach_nc_chosen_ints = choose_multiple_list(mach_nc_paths, "NetCDF file",
                                               null_action="perform Mach diagnostics on HDF5 files")
    if len(mach_nc_chosen_ints) > 0:
        # print(f"Loading {len(mach_nc_chosen_ints)} Mach NetCDF datasets...")
        mach_ds_list = [open_netcdf(mach_nc_paths[choice]) for choice in mach_nc_chosen_ints]
        steady_state_times = mach_ds_list[0].attrs['Steady state times']  # should be ms
    else:
        print("The following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=26))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file")
        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        mach_ds_list = []
        for hdf5_path in hdf5_chosen_list:  # TODO improve loading bar for many datasets

            print("\nOpening HDF5 file", repr(hdf5_path), "...")

            exp_params = get_exp_params(hdf5_path)
            run_name = exp_params["Run name"]
            lang_nc_path = make_path(lang_nc_folder, run_name, "nc")
            if not check_netcdf(lang_nc_path):
                print(f"No corresponding Langmuir diagnostics dataset NetCDF file found at {lang_nc_path}."
                      "This HDF5 file will be ignored.")
                continue
            lang_ds = open_netcdf(lang_nc_path)

            steady_state_ramps = detect_steady_state_ramps(lang_ds['n_e'], core_radius)
            steady_state_times = [lang_ds.coords['time'].isel(time=ramp - 1).item() for ramp in steady_state_ramps]

            linear_electron_temperature = linear_profile(lang_ds['T_e'],
                                                         steady_state_times)  # TODO fix lang_nc time


            # Get Isat data from HDF5 file and then process to obtain velocity
            mach_isat_da = get_mach_isat(hdf5_path, mach_face_bcs, mach_receptacles, mach_face_resistances)
            mach_ds = get_velocity_profiles(mach_isat_da, linear_electron_temperature, ion).assign_attrs(exp_params)

            # Attach experimental parameters at LAPD to Mach dataset attributes
            mach_ds = mach_ds.assign_attrs(exp_params)
            mach_ds = mach_ds.assign_attrs({"Steady state times": steady_state_times})  # should be ms

            mach_ds_list.append(mach_ds)
            if save_mach_ds:
                mach_ds_save_path = make_path(mach_nc_folder, run_name, "nc")
                write_netcdf(mach_ds, mach_ds_save_path)

    # Plot chosen diagnostics for each individual dataset
    """
    for plot_diagnostic in diagnostic_chosen_list:
        for dataset in datasets:
            try:
                line_time_diagnostic_plot(port_selector(dataset), plot_diagnostic, 'contour', steady_state_plateaus)
            except Exception as e:
                print(e)
    # """
    """
    for probe in range(len(mach_ds.port)):
        for variable in mach_ds.isel(port=probe):
            mach_ds.isel(port=probe)[variable].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf(robust=True)
            title = f"{variable} (Port {mach_ds.port[probe].item()})"
            plt.title(title)
            plt.show()
            linear_profile(mach_ds.isel(port=probe)[variable].mean(dim='shot', keep_attrs=True),
                           steady_state_times).squeeze().plot(x='x')
            plt.title(title)
            plt.show()
    # """

    # PLOT radial profiles of diagnostic (steady state time average), in color corresponding to first attribute,
    #    and in plot position on multiplot corresponding to second attribute
    # TODO assume all mach_ds have same number of ports and same diagnostics available, so only consider first dataset
    plot_diagnostic_list = mach_ds_list[0].keys()
    print("Plotting diagnostic comparisons...")
    for plot_diagnostic in plot_diagnostic_list:
        plot_line_diagnostic_by(mach_ds_list, plot_diagnostic, port_selector,
                                attribute=["Nominal discharge", "Nominal gas puff"])
        # TODO user select attribute(s) from menu
