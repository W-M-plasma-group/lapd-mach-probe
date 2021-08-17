import matplotlib.pyplot as plt
from setup import *
from getIsat import *

"""if __name__ == '__main__':"""

# Global parameters
sample_sec = (100 / 16 * 10 ** 6) ** -1 * u.s
hdf5_filename = "/Users/leo/Plasma code/HDF5/8-3500A.hdf5"

lapd_setup = setup_lapd(hdf5_filename)
isat = get_isat(hdf5_filename, sample_sec)
# isat[0].mean(dim='shot', keep_attrs=True).squeeze().plot.contourf()
# plt.show()
