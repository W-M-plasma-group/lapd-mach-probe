import numpy as np
from astropy import visualization
from astropy import units as u
import matplotlib.pyplot as plt
from radial import *

# Plotting options
import matplotlib
matplotlib.rcParams["figure.dpi"] = 150


def plot_line_diagnostic_by(mach_ds_list: list, plot_diagnostic, port_selector, attribute, tolerance=1/2, share_y=True):

    steady_state = mach_ds_list[0].attrs['Steady state times']

    attributes = np.atleast_1d(attribute)
    if len(attributes) > 2:
        raise ValueError("Cannot categorize line plots by more than two attributes")

    mach_ds_list_sorted = mach_ds_list  # not sorted yet
    for attr in attributes:
        try:
            mach_ds_list_sorted.sort(key=lambda d: d.attrs[attr])
        except KeyError:
            raise KeyError("Key error for key", repr(attr))
    outer_attr_values = [mach_ds.attrs[attributes[1]] for mach_ds in mach_ds_list_sorted]
    outer_attr_quantity = u.Quantity([value_safe(value) for value in outer_attr_values], unit_safe(outer_attr_values[0]))
    outer_attr_unique, outer_attr_indexes = np.unique(outer_attr_quantity, return_index=True) if len(attributes) == 2 else ([None], [0])
    outer_attr_bounds = np.append(outer_attr_indexes, len(outer_attr_values))

    visualization.quantity_support()
    plt.rcParams['figure.figsize'] = (4 + 4 * len(outer_attr_indexes), 6)
    fig, axes = plt.subplots(1, len(outer_attr_indexes), sharey="row")

    port_string = port_selector(mach_ds_list_sorted[0])[0]
    fig.suptitle(f"{plot_diagnostic}, {port_string}", size=18)
    for outer_index in range(len(outer_attr_unique)):
        outer_val = outer_attr_unique[outer_index]
        ax = np.atleast_1d(axes)[outer_index]

        datasets = mach_ds_list_sorted[outer_attr_bounds[outer_index]:outer_attr_bounds[outer_index + 1]]
        num_datasets = outer_attr_bounds[outer_index + 1] - outer_attr_bounds[outer_index]

        color_map = plt.cm.get_cmap("plasma")(np.linspace(0, 1, num_datasets))
        for inner_index in range(num_datasets):
            dataset = port_selector(datasets[inner_index])[1]  # TODO allow looping through multiple datasets returned
            inner_val = dataset.attrs[attributes[0]]

            if "shot" in dataset.dims:
                dataset = dataset.mean('shot', keep_attrs=True)

            linear_dimension = validate_dataset_dims(dataset.sizes)
            linear_da = dataset.squeeze()[plot_diagnostic]

            linear_da = linear_profile(linear_da, steady_state, mean=False)
            linear_da_mean = linear_da.mean('time', keep_attrs=True)
            # linear_da_std = linear_da.std('time', ddof=1)

            # Filters out sections of highly variable data - hardcoded
            max_variation = tolerance * np.abs(linear_da_mean.mean())
            # linear_da_calm = linear_da_mean.where(np.logical_and(linear_da_std < max_variation,
            #                                                      linear_da_std.notnull()))
            linear_da_smooth = linear_da_mean.where(
                np.abs(linear_da_mean.differentiate("x").differentiate("x")) < max_variation / 4)
            # TODO incorporate tolerance

            ax.plot(linear_da.coords[linear_dimension], -linear_da_smooth,  # TODO remove negative!
                    color=color_map[inner_index], label=str(inner_val))
            ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text((attribute[1] + ": " + str(outer_val) if len(attribute) == 2 else "")
                          + "\nColor: " + attribute[0])
        ax.legend()

    plt.tight_layout()
    plt.show()


# TODO redundant when merged with lapd-plasma-analysis
def value_safe(quantity_or_scalar):  # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar  # input is a dimensionless scalar with no value
    return val


def unit_safe(quantity_or_scalar):  # Get unit of quantity or scalar, if possible

    try:
        unit = quantity_or_scalar.unit
    except AttributeError:
        unit = None  # The input data is dimensionless
    return unit
