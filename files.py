import os
import warnings

import xarray as xr


def choose_list(choices, kind, location, add_new):
    print("The following " + kind + "s were found in the " + location + ":")
    print(*["  " + str(i + 1) + ":\t " + str(choices[i]) for i in range(len(choices))], sep="\n")
    prompt = "Type a number to select the corresponding " + kind
    if add_new:
        prompt += ", \n\tor 0 to create a new " + kind
    prompt += ": "
    return int(input(prompt))


def choose_multiple_list(choices, name, null_action=None):

    if len(choices) > 26:
        warnings.warn("More than 26 " + name + "s found. Only the first 26 are displayed.")
    print(*["  " + chr(97 + i) + ": " + choices[i] for i in range(len(choices[:26]))], sep="\n")
    prompt = "Input a string of letters to select the corresponding " + name + "s"
    if null_action is not None:
        prompt += ", \n\tor the empty string to " + null_action
    prompt += ": "
    selection_str = input(prompt).lower()

    if selection_str == "" and null_action is not None:
        return []
    if not selection_str.isalpha():
        raise ValueError("Selection " + repr(selection_str) + " is not only letters")

    return [ord(letter) - 97 for letter in selection_str if 0 <= ord(letter) - 97 < len(choices)]


# Ensure that the file path contains a Dataset
def check_netcdf(filename):
    try:
        xr.open_dataset(filename)
    except FileNotFoundError:
        return False
    return True


def open_netcdf(filename):
    # print("Opening NetCDF dataset file...")
    return xr.open_dataset(filename)


def write_netcdf(dataset, path):
    # print("Saving diagnostic dataset...")
    save_mode = 'a' if check_netcdf(path) else 'w'
    dataset.to_netcdf(path=path, mode=save_mode)


# Search the given directory and all subfolders for files of desired extension
def search_folder(directory, ext, limit=None) -> list:
    netcdf_files = []
    for path, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith("." + ext):
                netcdf_files.append(os.path.join(path, filename))
                if type(limit) == int and len(netcdf_files) >= limit:
                    return netcdf_files
    return netcdf_files


def ensure_directory(directory_path: str):
    head, tail = os.path.split(directory_path)
    if tail != "":
        name, ext = os.path.splitext(tail)
        if ext == "":
            warnings.warn("Adding final '/' to directory path " + repr(directory_path))
            directory_path += "/"
        else:
            raise ValueError("The path " + repr(directory_path) + " is not a directory path")
    if not os.path.isabs(directory_path):
        raise ValueError("Path must be absolute")
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    return directory_path


# Generate a file path from a folder, a filename (not a path), and an extension
def make_path(folder: str, name: str, ext: str) -> str:
    # full_path = os.path.join(folder, bimaxwellian_filename + ".nc")
    extension = ext if ext.startswith(".") else "." + ext
    return os.path.join(folder, name + extension)
