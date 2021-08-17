import matplotlib.pyplot as plt
from setup import *
from getIsat import *
from velocity import *

"""if __name__ == '__main__':"""

# Global parameters
sample_sec = (100 / 16 * 10 ** 6) ** -1 * u.s
hdf5_filename = "/Users/leo/Plasma code/HDF5/8-3500A.hdf5"
diagnostic_open_filename = "/Users/leo/Plasma code/lapd-plasma-analysis/diagnostic_dataset.nc"
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
save_isat = True
"""Set the use_existing_isat variable to True to open an existing Mach dataset, or False to create a new one."""
use_existing_mach = False
"""Set the save_isat variable to True to save a Mach dataset if a new one is created."""
save_mach = False
# End user settings

lapd_parameters = setup_lapd(hdf5_filename)

# Check if diagnostic data netCDF file with diagnostic data exists
# open_file_exists = True
diagnostic_dataset = xr.open_dataset(diagnostic_open_filename)

if use_existing_isat:
    print("Opening isat data file...")
    isat = xr.open_dataarray(isat_open_filename)
else:
    isat = get_isat(hdf5_filename, sample_sec)
    if save_isat:
        isat.to_netcdf(isat_save_filename)

"""  INCOMPLETE
if use_existing_mach:
    print("Opening Mach data file...")
    mach = xr.open_dataarray(isat_open_filename)
else:
    mach = get_velocity_profiles(hdf5_filename, sample_sec)
    if save_isat:
        isat.to_netcdf(isat_save_filename)
"""

# isat[0].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
# plt.show()

parallel_mach, perpendicular_mach, perpendicular_mach_fore, perpendicular_mach_aft, \
    parallel_velocity, perpendicular_velocity = get_velocity_profiles(isat, diagnostic_dataset['T_e'])

print(parallel_velocity)
parallel_velocity.mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
plt.show()
