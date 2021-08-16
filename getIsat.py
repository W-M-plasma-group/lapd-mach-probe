from pprint import pprint

import numpy as np
import xarray as xr
import astropy.units as u

from hdf5reader import *

# Note: This code is based heavily on similar code written by the same author for the LAPD plasma analysis repository.
# MAKE GET_ISWEEP_VSWEEP A NAMESPACE PACKAGE WITH GENERIC DATA TO ACCESS AND IMPORT HERE


def get_isat(filename):

    hdf5_file = open_hdf5(filename)
    x_round, y_round, shot_list = get_mach_xy(hdf5_file)
    xy_shot_ref, x, y = categorize_mach_xy(x_round, y_round, shot_list)

    print("Reading raw data and headers...")
    mach_data, mach_headers = get_mach_data_headers(hdf5_file)

    print("Decompressing mach data...")
    isat_array = scale_offset_decompress(mach_data, mach_headers)

    # Store six separate 4D arrays: x, y, shot number at position, frame number in shot (5D overall)
    isat_xy_shots_array = np.array([isat_face[xy_shot_ref] for isat_face in isat_array])

    hdf5_file.close()
    return isat_xy_shots_array, x, y


def get_mach_xy(hdf5_file):

    mach_data = structures_at_path(hdf5_file, "/Raw data + config/6K Compumotor")
    mach_path = (mach_data["Datasets"])[2]
    mach_motion = hdf5_file[mach_path]
    print("Mach data:", mach_motion)

    places = 1
    x_round = np.round(mach_motion['x'], decimals=places)
    y_round = np.round(mach_motion['y'], decimals=places)

    shot_list = np.argsort(mach_motion['Shot number'])

    return x_round, y_round, shot_list


def categorize_mach_xy(x_round, y_round, shot_list):  # Taken from LAPD getIVsweep.py categorize_xy function
    r"""
    Categorize shots by their x,y position. Returns a 3D list of shot numbers at each combination of unique x and y pos.

    Parameters
    ----------
    :param x_round: list
    :param y_round: list
    :param shot_list: list
    :return: list
    """

    x, x_loc = np.unique(x_round, return_inverse=True)
    y, y_loc = np.unique(y_round, return_inverse=True)
    x_length = len(x)
    y_length = len(y)

    # Determine if data is areal, radial, or scalar
    if x_length == 1 and y_length == 1:
        print("Only one position value. No plots can be made")
    elif x_length == 1:
        print("Only one unique x value. Will create radial plots along y dimension")
    elif y_length == 1:
        print("Only one unique y value. Will create radial plots along x dimension")

    # Can these be rewritten as NumPy arrays?

    # Store a reference to each shot number in a grid at x and y coordinates corresponding to unique x and y positions
    # For every shot index i, x_round[i] and y_round[i] give the x,y position of the shot taken at that index
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(len(shot_list)):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken

    """
    Should I rewrite getIVsweep.py in other project to make it able to read any data of user's choice, then use it
        to collect bias and current data in lapd-plasma-analysis and Isat in lapd-mach-probe?
    """

    return xy_shot_ref, x, y

    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING


def get_mach_data_headers(hdf5_file):

    # SIS crate data
    # sis_group = structures_at_path(hdf5_file, '/Raw data + config/SIS crate/')
    sis_data = structures_at_path(hdf5_file, '/Raw data + config/SIS crate/')['Datasets']

    # pprint(sis_data, width=120)

    # Mach probe has 6 faces; data are in slot numbers 12, 14, ..., 22 and corresponding headers in numbers 13, ..., 23
    mach_data_paths = [sis_data[12 + (2 * i)] for i in range(6)]
    mach_headers_paths = [sis_data[12 + (2 * i + 1)] for i in range(6)]

    # print(hdf5_file[mach_headers_paths[0]].dtype)
    # Converting entire structured datasets into NumPy arrays is slow?
    # mach_data_raw = [hdf5_file[path] for path in mach_data_paths]
    # mach_headers_raw = [hdf5_file[path] for path in mach_headers_paths]
    mach_data_raw = np.array([hdf5_file[path] for path in mach_data_paths])
    mach_headers_raw = np.array([hdf5_file[path] for path in mach_headers_paths])

    # print(mach_headers_raw.dtype)
    # print(mach_headers_raw['Scale'])

    # print("Shape of mach data array:", mach_data_raw.shape, "and headers array:", mach_headers_raw.shape)

    return mach_data_raw, mach_headers_raw


def scale_offset_decompress(data_raw, data_headers):  # change INPUT PARAMETERS and DIMENSION OF INPUT
    r"""
    Decompress raw data using the specified arrays of scales and offsets.
    Scale and offset arrays must have first dimension corresponding to the length of the input data
    (for example, the number of shots taken).

    Parameters
    ----------
    :param data_raw: array
    :param data_headers: structured array
    :return: decompressed data array
    """

    scales = data_headers['Scale']
    offsets = data_headers['Offset']

    if len(data_raw.shape) not in [2, 3]:
        raise ValueError("Only 2D or 3D arrays are currently supported for data decompression.")

    if data_raw.shape[:-1] != data_headers.shape:
        raise ValueError("Data and headers have incompatible shapes and cannot be broadcast together")

    # Reshape scales and offsets to have same dimensions as raw data, except with last dimension one to allow broadcast
    scales = scales.reshape(*data_raw.shape[:-1], 1)
    offsets = offsets.reshape(*data_raw.shape[:-1], 1)
        
    return data_raw * scales + offsets
