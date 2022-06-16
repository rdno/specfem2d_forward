#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plots all seismograms in the output_folder

Ridvan Orsvuran, 2022
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import obspy


def read_ascii_trace(filename: Path):
    """Reads SPECFEM2D ascii file and returns a obspy.Trace"""
    data = np.loadtxt(filename)
    # Find deltat
    dt = data[:, 0][1]-data[:, 0][0]
    try:
        net, sta, comp = filename.stem.split(".")
    except ValueError:
        net, sta, comp = "", "", ""

    stats = {"delta": dt, "network": net, "station": sta,
             "channel": comp, "b": data[0, 0]}
    return obspy.Trace(data[:, 1], stats)


def plot_all_seismograms(output_folder: Path, bandpass_range=None,
                         **extra_fig_opt):
    """Assumes P/SV Simulation"""
    bxx_files = sorted(output_folder.glob("*BXX.semd"))
    bxz_files = sorted(output_folder.glob("*BXZ.semd"))
    n_stations = len(bxx_files)

    fig_opt = {'sharex': True, 'sharey': True,
               'figsize': (20, 2*n_stations)}
    fig_opt.update(extra_fig_opt)

    fig, axes = plt.subplots(nrows=n_stations, ncols=2,
                             **fig_opt)

    for i, bxx_file in enumerate(bxx_files):
        tr = read_ascii_trace(bxx_file)
        if bandpass_range:
            fmin, fmax = bandpass_range
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)
        axes[i, 0].plot(tr.times()+tr.stats.b, tr.data, "k", label=tr.id)
        axes[i, 0].legend()
        axes[i, 0].set_xlim(tr.stats.b, tr.times()[-1]+tr.stats.b)

    for i, bxz_file in enumerate(bxz_files):
        tr = read_ascii_trace(bxz_file)
        if bandpass_range:
            fmin, fmax = bandpass_range
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)
        axes[i, 1].plot(tr.times()+tr.stats.b, tr.data, "k", label=tr.id)
        axes[i, 1].legend()
    return fig, axes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Seismograms")
    parser.add_argument("output_folder",
                        help="folder that contains the folder")
    parser.add_argument("-o", "--output_file", help="output_file for figure.",
                        default=None)
    parser.add_argument("-f", "--bandpass", nargs=2, type=float, default=None,
                        metavar=("min_freq", "max_freq"),
                        help="bandpass filter")

    # Reading
    args = parser.parse_args()

    # Plotting
    fig, axes = plot_all_seismograms(Path(args.output_folder),
                                     bandpass_range=args.bandpass)
    fig.tight_layout()
    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()
