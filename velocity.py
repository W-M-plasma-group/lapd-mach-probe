import numpy as np
import xarray as xr
import astropy.units as u


def get_velocity_profiles(isat_xarray, electron_temperature_xarray):
    r"""

    Parameters
    ----------
    :param isat_xarray:
    :param electron_temperature_xarray:
    :return:
    """

    """
        
        Model of Mach probe faces (perfect octagon)
                       ___________
            |         /           \ 
    fore    |    3  /               \  4
            |      |                 |
    cathode |   2  |                 |  5
            |      |                 |
    aft     |    1  \               /  6
            |         \___________/ 
    
    """

    """CONSTANTS AND DESCRIPTIONS ARE TAKEN FROM MATLAB CODE WRITTEN BY CONOR PERKS"""
    # Mach probe calculation constants
    magnetization_factor = 0.5  # Mag. factor value from Hutchinson's derivation incorporating diamagnetic drift
    alpha_fore = np.pi / 4 * u.rad  # [rad] Angle the face in fore direction makes with B-field
    alpha_aft = np.pi / 4 * u.rad  # [rad] Angle the face in aft direction makes with B-field

    # Velocity calculation constants
    ion_mass = 6.6464764 * 10 ** -27 * u.kg  # Ion mass
    ion_temperature = 1 * u.eV  # Approximate Ion temperature

    """Parallel Mach number"""
    print("Calculating Mach numbers...")
    parallel_mach = magnetization_factor * np.log(isat_xarray.sel(face=5) / isat_xarray.sel(face=2))
    print(" * Parallel Mach number found ")

    """Perpendicular Mach number"""
    mach_correction_fore = magnetization_factor * np.log(isat_xarray.sel(face=6) / isat_xarray.sel(face=3))
    mach_correction_aft = magnetization_factor * np.log(isat_xarray.sel(face=4) / isat_xarray.sel(face=1))

    perpendicular_mach_fore = (parallel_mach - mach_correction_fore) * np.cos(alpha_fore)
    perpendicular_mach_aft = (parallel_mach - mach_correction_aft) * np.cos(alpha_aft)
    perpendicular_mach = xr.concat([perpendicular_mach_fore, perpendicular_mach_aft], 'temp').mean('temp')
    print(" * Perpendicular Mach number found ")

    """Steady-state velocity profiles"""
    print("Generating velocity profiles...")
    # MATLAB note: "Note that M=v/C_s where C_s = sqrt((T_e+T_i)/M_i), but we will assume that T_i~1ev"

    # TODO Get rid of extra 'plateau' dimension somehow
    # Reshape mach-to-velocity conversion factor (from electron temperature) to make compatible with Mach number data
    # mach_to_velocity = np.sqrt(electron_temperature_xarray / ion_mass).expand_dims('temp', axis=-1)
    mach_to_velocity = np.sqrt(electron_temperature_xarray / ion_mass)

    # Save units of mach-number-to-velocity data to compute conversion factor for final result
    mach_to_velocity_conversion = np.sqrt(1 * (T_e_units := u.eV) / (m_i_units := u.kg)).to(u.cm/u.s).value  # see above

    # print("Parallel mach shape is", parallel_mach.shape, "and mach to velocity shape is", mach_to_velocity.shape)
    # parallel_velocity = (parallel_mach * mach_to_velocity).squeeze('temp') * mach_to_velocity_conversion
    parallel_velocity = parallel_mach * mach_to_velocity * mach_to_velocity_conversion
    parallel_velocity.attrs['units'] = str(u.cm/u.s)
    # perpendicular_velocity = (perpendicular_mach * mach_to_velocity).squeeze('temp') * mach_to_velocity_conversion
    perpendicular_velocity = perpendicular_mach * mach_to_velocity * mach_to_velocity_conversion
    perpendicular_velocity.attrs['units'] = str(u.cm / u.s)

    # TODO return these values as one Dataset instead of many DataArrays
    return parallel_mach, perpendicular_mach, perpendicular_mach_fore, perpendicular_mach_aft, \
        parallel_velocity, perpendicular_velocity
