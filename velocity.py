import numpy as np
import xarray as xr
import astropy.units as u

from plasmapy.particles import Particle


def get_velocity_profiles(mach_isat_da, electron_temperature_da , ion_type):
    r"""

    Parameters
    ----------
    :param mach_isat_da:
    :param electron_temperature_da:
    :param ion_type:
    :return:
    """

    """
        
        Model of Mach probe faces (actual probe is perfect octagon)
                            ___________
                 |         /           \ 
        fore     |    3  /               \  4
                 |      |                 |
    (<- Cathode) |   2  |                 |  5            <----  B-field
                 |      |                 |
        aft      |    1  \               /  6
                 |         \___________/ 
    
    """ # noqa

    """
    Diagnostic DataArray structure
    - Dims: port, x (cm), y (cm), time (ms)
    - Coordinates: plateau (with time, starting from 1); z (cm, with port)
    - 
    """

    """CONSTANTS AND DESCRIPTIONS ARE TAKEN FROM MATLAB CODE WRITTEN BY CONOR PERKS"""
    # Mach probe calculation constants
    magnetization_factor = 0.5  # Mag. factor value from Hutchinson's derivation incorporating diamagnetic drift
    alpha_fore = np.pi / 4 * u.rad  # [rad] Angle the face in fore direction makes with B-field
    alpha_aft = np.pi / 4 * u.rad  # [rad] Angle the face in aft direction makes with B-field

    # Velocity calculation constants

    # ion_mass = 6.6464764e-27 * u.kg  # Ion mass
    ion_mass = Particle(ion_type).mass
    # ion_temperature = 1 * u.eV  # Approximate Ion temperature

    mach_to_velocity = np.sqrt(electron_temperature_da / ion_mass).sortby("port")  # Local plasma sound speed in sqrt(eV/kg)

    # Factor to convert velocity (using above factor) from sqrt(eV / kg) to cm / s
    mach_to_velocity_units = np.sqrt(1 * u.eV / u.kg).to(u.cm / u.s).value
    # MATLAB note: "Note that M=v/C_s where C_s = sqrt((T_e+T_i)/M_i), but we will assume that T_i~1ev"

    # print("Calculating Mach numbers...")
    
    """Parallel Mach number"""
    parallel_mach = magnetization_factor * np.log(mach_isat_da.sel(face=5) / mach_isat_da.sel(face=2)).sortby("port")
    # print(" * Parallel Mach number found ")

    """Parallel steady-state velocity profiles"""
    # print(" * Generating parallel velocity profiles...")
    langmuir_with_mach_ports = mach_to_velocity.assign_coords(port=parallel_mach.port)
    mach_to_velocity = mach_to_velocity.reindex_like(langmuir_with_mach_ports, method="nearest", tolerance=5)
    parallel_velocity = parallel_mach * mach_to_velocity * mach_to_velocity_units
    parallel_velocity.attrs['units'] = str(u.cm / u.s)

    mach_velocities = xr.Dataset({"Parallel Mach number": parallel_mach,
                                  "Parallel velocity": parallel_velocity})

    """Perpendicular Mach number"""
    if np.isin(np.array([1, 3, 4, 6]), mach_isat_da.face).all():
        mach_correction_fore = magnetization_factor * np.log(mach_isat_da.sel(face=6) / mach_isat_da.sel(face=3))
        mach_correction_aft = magnetization_factor * np.log(mach_isat_da.sel(face=4) / mach_isat_da.sel(face=1))

        perpendicular_mach_fore = (parallel_mach - mach_correction_fore) * np.cos(alpha_fore)
        perpendicular_mach_aft = (parallel_mach - mach_correction_aft) * np.cos(alpha_aft)
        perpendicular_mach = xr.concat([perpendicular_mach_fore, perpendicular_mach_aft], 'location').mean('location')
        # print(" * Perpendicular Mach number found ")

        """Perpendicular steady-state velocity profiles"""
        # print(" * Generating perpendicular velocity profiles...")
        perpendicular_velocity = perpendicular_mach * mach_to_velocity * mach_to_velocity_units
        perpendicular_velocity.attrs['units'] = str(u.cm / u.s)

        mach_velocities = mach_velocities.assign({"Perpendicular Mach number": perpendicular_mach,
                                                  "Perpendicular fore Mach number": perpendicular_mach_fore,
                                                  "Perpendicular aft Mach number": perpendicular_mach_aft,
                                                  "Perpendicular velocity": perpendicular_velocity})

    return -mach_velocities  # A positive velocity is directed *away* from the cathode now


def in_core(pos_list, core_radius):
    return [abs(pos) < core_radius.to(u.cm).value for pos in pos_list]


def detect_steady_state_ramps(density: xr.DataArray, core_radius):
    core_density = density.where(np.logical_and(*in_core([density.x, density.y], core_radius)), drop=True)
    core_density_time = core_density.isel(port=0).mean(['x', 'y']).squeeze()
    threshold = 0.9 * core_density_time.max()
    start_index = (core_density_time > threshold).argmax().item() + 1
    # TODO improve search for last True index
    end_index = core_density_time.sizes['time'] - (
            core_density_time.reindex(time=core_density_time.time[::-1]) > threshold).argmax().item()
    return start_index, end_index
