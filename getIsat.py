import numpy as np
import xarray as xr
import astropy.units as u

from hdf5reader import *

# Note: This code is based heavily on similar code written by the same author for the LAPD plasma analysis repository.


def get_isat(filename):

    hdf5_file = open_hdf5(filename)
    x_round, y_round, shot_list = get_mach_xy(hdf5_file)
    xy_shot_ref, x, y = categorize_mach_xy(x_round, y_round, shot_list)

    isat_array = None

    return isat_array, x, y


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
        print("Only one unique x value. Will only consider y position")
    elif y_length == 1:
        print("Only one unique y value. Will only consider x position")

    # Can these be rewritten as NumPy arrays?

    # Store a reference to each shot number in a grid at x and y coordinates corresponding to unique x and y positions
    # For every shot index i, x_round[i] and y_round[i] give the x,y position of the shot taken at that index
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(len(shot_list)):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken

    return xy_shot_ref, x, y

    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING

# TODO Should I rewrite getIVsweep.py in other project to make it able to read any data of user's choice, then use it
#  to collect bias and current data in lapd-plasma-analysis and Isat in lapd-mach-probe?
