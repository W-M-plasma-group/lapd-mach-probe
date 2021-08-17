import numpy as np
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
    ALL CONSTANTS AND DESCRIPTIONS ARE TAKEN FROM MATLAB CODE WRITTEN BY CONOR PERKS
    
        Model of Mach probe faces (perfect octagon)
                        ________
            |         /         \ 
    fore    |    3  /             \  4
            |      |               |
    cathode |   2  |               |  5
            |      |               |
    aft     |    1  \             /  6
            |         \_________/ 
    
    """

    # Mach probe calculation constants
    magnetization_factor = 0.5  # Mag. factor value from Hutchinson's derivation incorporating diamagnetic drift
    alpha_fore = np.pi / 4 * u.rad  # [rad] Angle the face in fore direction makes with B-field
    alpha_aft = np.pi / 4 * u.rad  # [rad] Angle the face in aft direction makes with B-field

    # Velocity calculation constants
    ion_mass = 6.6464764 * 10 ** -27 * u.kg  # Ion mass
    ion_temperature = 1 * u.eV  # Approximate Ion temperature

    """Parallel Mach number"""

    parallel_mach = magnetization_factor * np.log(isat_xarray.isel(face=5) / isat_xarray.isel(face=2))

    """Perpendicular Mach number"""

    mach_correction_fore = magnetization_factor * np.log(isat_xarray.isel(face=6) / isat_xarray.isel(face=3))
    mach_correction_aft = magnetization_factor * np.log(isat_xarray.isel(face=4) / isat_xarray.isel(face=1))

    perpendicular_mach_fore = (parallel_mach - mach_correction_fore) * np.cos(alpha_fore)
    perpendicular_mach_aft = (parallel_mach - mach_correction_aft) * np.cos(alpha_aft)
    perpendicular_mach = np.mean([perpendicular_mach_fore, perpendicular_mach_aft], axis=0)

    """Steady-state velocity profiles"""

    # MATLAB note: "Note that M=v/C_s where C_s = sqrt((T_e+T_i)/M_i), but we will assume that T_i~1ev"
    # parallel_velocity = parallel_mach * np.sqrt()
