#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stripped down version of sair-plot-model
Plots binary model files

Ridvan Orsvuran 2021-2022
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import argparse


def read_data(filename):
    nbytes = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        f.seek(0)
        n = np.fromfile(f, dtype='int32', count=1)[0]

        if n == nbytes-8:
            f.seek(4)
            data = np.fromfile(f, dtype='float32')
            return data[:-1]
        else:
            f.seek(0)
            data = np.fromfile(f, dtype='float32')
    return data


def data_filename(folder, param, rank=0):
    return os.path.join(folder, f"proc{rank:06d}_{param}.bin")


def read_data_from_folder(folder, param, rank=0):
    return read_data(data_filename(folder, param, rank))


def get_nproc(folder, param="x"):
    return len(glob.glob(os.path.join(folder, f"proc??????_{param}.bin")))


def read_all_from_folder(folder, param, nproc=None):
    if nproc is None:
        nproc = get_nproc(folder, param)
    if nproc == 0:
        raise Exception(f"No '{param}' data found in folder: {folder}")

    for iproc in range(nproc):
        if iproc == 0:
            data = read_data_from_folder(folder, param, iproc)
        else:
            new_data = read_data_from_folder(folder, param, iproc)
            data = np.append(data, new_data, axis=0)
    return data


def grid(x, y, z, resX=500, resY=500):
    """
    Converts 3 column data to grid
    """
    from scipy.interpolate import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='linear')

    return X, Y, Z


def plot_bin(x, y, z):
    vmax = np.max(z)
    vmin = np.min(z)

    cmap = CMAP

    X, Y, Z = grid(x, y, z)

    fig, ax = plt.subplots()

    im = ax.imshow(Z, vmax=vmax, vmin=vmin,
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap,
                   origin='lower')

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, ax=ax)

    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])

    xticks = np.linspace(min(x), max(x), 6)
    yticks = np.linspace(min(y), max(y), 6)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    unit = "m"
    zlabel = "Z"

    ax.set_xlabel("X ({})".format(unit))
    ax.set_ylabel("{} ({})".format(zlabel, unit))

    xticklabels = xticks + 0
    yticklabels = yticks + 0

    ax.set_xticklabels(["{:.1f}".format(lbl) for lbl in xticklabels])
    ax.set_yticklabels(["{:.1f}".format(lbl) for lbl in yticklabels])

    plt.tight_layout()

    return fig, ax


model_labels = {
    "vp": "P-Wave speed ($m/s$)",
    "vs": "S-Wave speed ($m/s$)",
    "rho": "Density ($kg/m^3$)",
    "Qmu": "$Q_\mu$",           # NOQA
    "Qkappa": "$Q_\kappa$"      # NOQA
}


def plot_model(data_folder, model_param):
    x = read_all_from_folder(data_folder, "x")
    z = read_all_from_folder(data_folder, "z")
    data = read_all_from_folder(data_folder, model_param)

    plot_bin(x, z, data)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot model")
    parser.add_argument('data_folder')
    parser.add_argument('parameter', choices=("vp", "vs", "rho", "Qmu", "Qkappa"))
    parser.add_argument("-c", "--colormap", help="colormap", default=None)

    args = parser.parse_args()
    CMAP = args.colormap or "bwr_r"
    plot_model(args.data_folder, args.parameter)

