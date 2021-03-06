import numpy as np
import astropy.units as u

from hdf5reader import *


def setup_lapd(filename, in_units=(u.Torr, u.gauss, u.A)):

    print("Setting up HDF5 file...")
    setup_file = open_hdf5(filename)

    # Note: hard-coded indices for relevant data
    # print(list(setup_file['/MSI/Gas pressure/'].attrs.items()))
    gas_pressure = np.array([list(row) for row in get_gas_pressure(setup_file)])
    mean_fill_pressure = np.mean(gas_pressure[..., 4])

    magnetic_field = np.array([list(row) for row in get_magnetic_field(setup_file)])
    mean_peak_field = np.mean(magnetic_field[..., 3])

    discharge = np.array([list(row) for row in get_discharge(setup_file)])
    mean_discharge = np.mean(discharge[..., 4])

    return {"Fill pressure": (mean_fill_pressure * u.Torr).to(in_units[0]),
            "Peak field": (mean_peak_field * u.gauss).to(in_units[1]),
            "Discharge": (mean_discharge * u.A).to(in_units[2])}              # in MATLAB-given units


def get_gas_pressure(file):
    return item_at_path(file, '/MSI/Gas pressure/Gas pressure summary/')


def get_magnetic_field(file):
    return item_at_path(file, '/MSI/Magnetic field/Magnetic field summary/')


def get_discharge(file):
    return item_at_path(file, '/MSI/Discharge/Discharge summary/')
