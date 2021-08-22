import numpy as np
import xarray as xr
import astropy.units as u


def get_velocity_profiles(isat_xarray, electron_temperature_xarray, experimental_parameters):
    r"""

    Parameters
    ----------
    :param isat_xarray:
    :param electron_temperature_xarray:
    :param experimental_parameters:
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
    # ion_temperature = 1 * u.eV  # Approximate Ion temperature

    mach_velocity = xr.Dataset()

    """Parallel Mach number"""
    print("Calculating Mach numbers...")
    parallel_mach = magnetization_factor * np.log(isat_xarray.sel(face=5) / isat_xarray.sel(face=2))
    print(" * Parallel Mach number found ")

    """Perpendicular Mach number"""
    mach_correction_fore = magnetization_factor * np.log(isat_xarray.sel(face=6) / isat_xarray.sel(face=3))
    mach_correction_aft = magnetization_factor * np.log(isat_xarray.sel(face=4) / isat_xarray.sel(face=1))

    perpendicular_mach_fore = (parallel_mach - mach_correction_fore) * np.cos(alpha_fore)
    perpendicular_mach_aft = (parallel_mach - mach_correction_aft) * np.cos(alpha_aft)
    perpendicular_mach = xr.concat([perpendicular_mach_fore, perpendicular_mach_aft], 'location').mean('location')
    print(" * Perpendicular Mach number found ")

    """Steady-state velocity profiles"""
    print("Generating velocity profiles...")
    # MATLAB note: "Note that M=v/C_s where C_s = sqrt((T_e+T_i)/M_i), but we will assume that T_i~1ev"

    mach_to_velocity = np.sqrt(electron_temperature_xarray / ion_mass)  # Local plasma Mach number value in cm/s

    # Factor to convert velocity (using above factor) from sqrt(eV / kg) to cm / s
    mach_to_velocity_units = np.sqrt(1 * (T_e_units := u.eV) / (m_i_units := u.kg)).to(u.cm / u.s).value

    parallel_velocity = parallel_mach * mach_to_velocity * mach_to_velocity_units
    parallel_velocity.attrs['units'] = str(u.cm / u.s)
    perpendicular_velocity = perpendicular_mach * mach_to_velocity * mach_to_velocity_units
    perpendicular_velocity.attrs['units'] = str(u.cm / u.s)

    return xr.Dataset({"Parallel Mach number": parallel_mach,
                       "Perpendicular Mach number": perpendicular_mach,
                       "Perpendicular fore Mach number": perpendicular_mach_fore,
                       "Perpendicular aft Mach number": perpendicular_mach_aft,
                       "Parallel velocity": parallel_velocity,
                       "Perpendicular velocity": perpendicular_velocity},
                      attrs=experimental_parameters)
