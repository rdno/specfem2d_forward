#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write model values into specfem2D binary files according to some rules.

Ridvan Orsvuran 2019-2021
"""

import numpy as np

from scipy import interpolate

import argparse
import glob
import os

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD           # NOQA
    rank = comm.Get_rank()          # NOQA
    size = comm.Get_size()          # NOQA
except ImportError:
    comm = None
    rank = 0
    size = 1

# specfem2d IO


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


def write_data(data, filename):
    n = np.array([4*len(data)], dtype='int32')
    data = np.array(data, dtype='float32')

    with open(filename, 'wb') as f:
        n.tofile(f)
        data.tofile(f)
        n.tofile(f)


def data_filename(folder, param, rank=0):
    return os.path.join(folder, f"proc{rank:06d}_{param}.bin")


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


def read_data_from_folder(folder, param, rank=0):
    return read_data(data_filename(folder, param, rank))


def write_data_to_folder(data, folder, param, rank=0):
    return write_data(data, data_filename(folder, param, rank))

# General functions


def get_typed_args(args, types):
    typed_args = []
    for arg, t in zip(args, types):
        try:
            typed_args.append(t(arg))
        except ValueError:
            raise Exception(f"{arg} is not in correct type: {t}")
    return typed_args


def get_model_bounds(data_folder):
    """Returns startx, startz, width, height"""
    x = read_all_from_folder(data_folder, "x")
    z = read_all_from_folder(data_folder, "z")
    startx = min(x)
    startz = min(z)
    width = max(x) - startx
    height = max(z) - startz
    return startx, startz, width, height


# Homogeneous Model


def write_homogenenous(value, param, data_folder):
    x = read_data_from_folder(data_folder, "x", rank)
    values = np.ones_like(x) * value
    write_data_to_folder(values, data_folder, param, rank)


# Checkerboard model


def checkerboard_model(x, y, v, t, width, height):
    g = np.sin(np.pi * x / width) * np.sin(np.pi * y / height)
    return v + (t * g + 1)


def write_checkerboard(nx, ny, mean, spread, param, data_folder):
    _, _, width, height = get_model_bounds(data_folder)
    x = read_data_from_folder(data_folder, "x", rank)
    z = read_data_from_folder(data_folder, "z", rank)
    data = checkerboard_model(x, z, mean, spread, width / nx, height / ny)
    write_data_to_folder(data, data_folder, param, rank)


# Model from an image


def get_values(image_file):
    from PIL import Image
    im = Image.open(image_file)
    pix = im.load()
    values = np.zeros(im.size)
    height = im.size[1]
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            try:
                values[i, j] = (pix[i, height - j - 1][0]) / 255
            except TypeError:  # grayscale image
                values[i, j] = (pix[i, height - j - 1]) / 255
    return values


def interp(values, startx, starty, width, height):
    mw, mh = values.shape
    mx = np.arange(startx, startx + width, width / (mw))
    my = np.arange(starty, starty + height, height / (mh))
    b = interpolate.RectBivariateSpline(mx, my, values)
    return b


def image_to_model(image_file, mean, spread, param, data_folder):
    values = get_values(image_file)
    startx, startz, width, height = get_model_bounds(data_folder)
    b = interp(values, startx, startz, width, height)
    x = read_data_from_folder(data_folder, "x", rank)
    z = read_data_from_folder(data_folder, "z", rank)
    data = b.ev(x, z)
    data = (data - 128 / 255) * 2 * spread + mean
    write_data_to_folder(data, data_folder, param, rank)


# Layercake model


def read_layercake_model(filename):
    model = []
    with open(filename) as f:
        for line in f:
            depth, val = line.split()
            model.append((float(depth), float(val)))
    return model


def layercake(filename, param, data_folder):
    startx, startz, width, height = get_model_bounds(data_folder)
    z = read_data_from_folder(data_folder, "z", rank)
    values = np.zeros(z.shape)
    depths = height - z
    model = read_layercake_model(filename)

    for i in range(len(model) - 1):
        values[
            np.logical_and(depths >= model[i][0], depths <= model[i + 1][0])
        ] = model[i][1]

    values[depths > model[-1][0]] = model[-1][1]
    write_data_to_folder(values, data_folder, param, rank)


def get_layercake_val(model, depth):
    for layer, value in model[::-1]:
        if depth >= layer:
            return value
    return value


def layercake_smooth(filename, sigma, param, data_folder):
    startx, startz, width, height = get_model_bounds(data_folder)
    zs = read_data_from_folder(data_folder, "z", rank)
    depths = height - zs
    model = read_layercake_model(filename)
    dx = sigma / 100
    new_zs = np.arange(np.min(depths) - 4 * sigma, np.max(depths) + 4 * sigma, dx)
    smoothed = np.zeros_like(depths)
    rvalues = np.array([get_layercake_val(model, z) for z in new_zs])
    A = 1.0 / (sigma * np.sqrt(2 * np.pi))
    for i, d in enumerate(depths):
        g = A * np.exp(-0.5 * ((new_zs - d) / (sigma)) ** 2)
        smoothed[i] = np.sum(g * rvalues) * dx
    write_data_to_folder(smoothed, data_folder, param, rank)


def layercake_smooth_plus_checkerboard(filename, sigma, nx, ny, mean,
                                       spread, param, data_folder):
    startx, startz, width, height = get_model_bounds(data_folder)
    xs = read_data_from_folder(data_folder, "x", rank)
    zs = read_data_from_folder(data_folder, "z", rank)
    depths = height - zs
    model = read_layercake_model(filename)
    dx = sigma / 100
    new_zs = np.arange(np.min(depths) - 4 * sigma, np.max(depths) + 4 * sigma, dx)
    smoothed = np.zeros_like(depths)
    rvalues = np.array([get_layercake_val(model, z) for z in new_zs])
    A = 1.0 / (sigma * np.sqrt(2 * np.pi))
    for i, d in enumerate(depths):
        g = A * np.exp(-0.5 * ((new_zs - d) / (sigma)) ** 2)
        smoothed[i] = np.sum(g * rvalues) * dx

    checkers = checkerboard_model(xs, zs, mean, spread, width / nx, height / ny)
    smoothed += checkers
    write_data_to_folder(smoothed, data_folder, param, rank)


if __name__ == "__main__":
    model_params = ["rho", "vp", "vs", "Qmu", "Qkappa"]
    model_types = {
        "homogeneous": write_homogenenous,
        "checkerboard": write_checkerboard,
        "image": image_to_model,
        "layercake": layercake,
        "smooth_layercake": layercake_smooth,
        "smooth_layercake_plus_checkerboard": layercake_smooth_plus_checkerboard
    }
    model_args = {
        "homogeneous": [("value", float)],
        "checkerboard": [("nx", int), ("ny", int), ("mean", float), ("spread", float)],
        "image": [("image_file", str), ("mean", float), ("spread", float)],
        "layercake": [("model_file", str)],
        "smooth_layercake": [("model_file", str), ("sigma", float)],
        "smooth_layercake_plus_checkerboard": [("model_file", str), ("sigma", float), ("nx", int), ("ny", int), ("mean", float), ("spread", float)],
    }

    parser = argparse.ArgumentParser(description="Generate model")

    for model_type in model_types.keys():
        metavar, types = zip(*model_args[model_type])
        arg_type = types[0] if len(metavar) == 1 else None
        for model_param in model_params:
            parser.add_argument(
                f"--{model_type}-{model_param}",
                metavar=metavar,
                nargs=len(metavar),
                type=arg_type,
                help=f"{model_type.title()} {model_param.title()} Model",
            )

    parser.add_argument("data_folder", help="data_folder")
    args = parser.parse_args()

    for model_type, model_func in model_types.items():
        metavar, types = zip(*model_args[model_type])
        for model_param in model_params:
            arg = getattr(args, f"{model_type}_{model_param}")
            if arg is not None:
                func_args = get_typed_args(arg, types)
                model_func(*func_args, param=model_param, data_folder=args.data_folder)
