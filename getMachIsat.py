import numpy as np
import xarray as xr
import astropy.units as u
from bapsflib import lapd
from bapsflib.lapd.tools import portnum_to_z as port_to_z

# Note: This code is based on getIVsweep.py in the lapd-plasma-analysis repository.
# MAKE GET_ISWEEP_VSWEEP A NAMESPACE PACKAGE WITH GENERIC DATA TO ACCESS AND IMPORT HERE?

# # BAPSFLIB parameters
# format of mach_face_bcs = [{probe1face1: (b, c), probe1face2: (b, c)},
#                             probe2face1: (b, c), probe2face2: (b, c)}]
# mach_face_bcs = [{2: (3, 1), 5: (3, 2)},
#                  {2: (3, 3), 5: (3, 4)}]
# mach_receptacles = [3, 4]
# mach_face_resistances = [{2: 14.9, 5: 15.0},
#                          {2: 14.9, 5: 15.0}]


def get_mach_isat(filename, mach_bcs, mach_receptacles, resistances):

    # TODO add function definition
    r"""

    :param filename:
    :param mach_bcs:
    :param mach_receptacles:
    :param resistances:
    :return:
    """

    lapd_file = lapd.File(filename)
    run_name = lapd_file.info['run name']

    # mach_bcs should be a list of dictionaries. Each dictionary entry pairs a face number with a board, channel tuple.
    # One dictionary corresponds to one probe, with its associated faces and board, channel tuples.
    isat_datas = [{face_num: lapd_file.read_data(*probe_bcs[face_num], silent=True) for face_num in probe_bcs}
                  for probe_bcs in mach_bcs]
    # print(isat_datas[0][2]['signal'].shape)

    mach_motor_datas = [lapd_file.read_controls([('6K Compumotor', receptacle)]) for receptacle in mach_receptacles]
    lapd_file.close()
    ports = np.array([motor_data.info['controls']['6K Compumotor']['probe']['port'] for motor_data in mach_motor_datas])
    # NOTE: Assume mach motor datas from 6K Compumotor are identical, and only consider first one
    positions, num_positions, shots_per_position = get_shot_positions(mach_motor_datas[0])

    isat_da = to_mach_isat_da(isat_datas, positions, shots_per_position, ports).rename(run_name)
    isat_da = to_real_mach_isat_units(isat_da, resistances)

    # Subtract out DC current offset on each face
    isat_offsets = isat_da[..., -2000:].quantile(0.25, dim="time")  # TODO should this be mean?
    isat_da -= isat_offsets
    # Drop negative values for current
    # Should not be negative in first place, so if needed for large areas, data may have been skewed/is unreliable
    isat_da = isat_da.where(isat_da > 0)

    return isat_da


def get_shot_positions(isat_motor_data):

    num_shots = len(isat_motor_data['shotnum'])
    shot_positions = np.round(isat_motor_data['xyz'], 1)

    z_positions = shot_positions[:, 2]
    # if np.amin(z_positions) != np.amax(z_positions):
    #     raise ValueError("Varying z-position when only x and/or y variation expected")
    # save z-position for later? Shouldn't need to, because hard to accidentally vary port
    positions = np.unique(shot_positions[:, :2], axis=0)  # list of all unique (x, y) positions
    num_positions = len(positions)
    if num_shots % num_positions != 0:
        raise ValueError("Number of Mach measurements " + str(num_shots) +
                         " does not evenly divide into " + str(num_positions) + " unique positions")
    shots_per_position = int(num_shots // num_positions)

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.amax(xy_at_positions, axis=1) == np.amin(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping Mach probe data by position")

    return positions, num_positions, shots_per_position


def to_real_mach_isat_units(isat_da, resistances):
    for probe in range(len(resistances)):
        for face in resistances[probe]:
            resistance = resistances[probe][face]
            isat_da[probe].loc[{"face": face}] *= resistance
    return isat_da


def to_mach_isat_da(isat_datas, positions, shots_per_position, ports):
    """

    :param isat_datas:
    :param shots_per_position:
    :param positions:
    :param ports:
    :return:
    """
    # [{face_num: isat_data for face_num in probe_bcs} for probe_bcs in mach_bcs]

    faces = sorted({face for probe in isat_datas for face in probe})
    x_pos = np.unique(positions[:, 0])
    y_pos = np.unique(positions[:, 1])
    # ports already given
    test_isat = isat_datas[0][list(isat_datas[0].keys())[0]]
    num_frames = test_isat['signal'].shape[-1]
    dt = test_isat.dt
    port_z = np.array([port_to_z(port).to(u.cm).value for port in ports])

    # isat_signals = [np.concatenate([isat_data['signal'][np.newaxis, ...] for isat_data in isat_datas], axis=0)]
    isat_signals_shape = (len(x_pos), len(y_pos), shots_per_position, num_frames)
    isat_signals = [{face: probe_isat_data['signal'].reshape(isat_signals_shape)
                     for face, probe_isat_data in probe_data.items()}
                    for probe_data in isat_datas]

    empty_isat_array = np.full((len(ports), len(faces), *isat_signals_shape), np.nan)
    isat_da = xr.DataArray(data=empty_isat_array,
                           dims=['port', 'face', 'x', 'y', 'shot', 'time'],
                           coords=(('port', ports),
                                   ('face', faces),
                                   ('x', x_pos, {"units": str(u.cm)}),
                                   ('y', y_pos, {"units": str(u.cm)}),
                                   ('shot', np.arange(shots_per_position) + 1),
                                   ('time', np.arange(num_frames) * dt.to(u.ms).value, {"units": str(u.ms)}))
                           ).assign_coords({'z': ('port', port_z)})
    for probe in range(len(isat_signals)):
        for face in isat_signals[probe]:
            isat_da[probe].loc[{"face": face}] = isat_signals[probe][face]

    return isat_da
