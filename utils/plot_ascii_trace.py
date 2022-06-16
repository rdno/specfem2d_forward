#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ridvan Orsvuran, 2022

Simple two column ascii file plottter
"""
import obspy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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


def plot_trace(tr: obspy.Trace):
    fig, ax = plt.subplots()
    ax.plot(tr.times()+tr.stats.b, tr.data, "k")
    return fig, ax


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Seismogram")
    parser.add_argument("seismogram", help="seismogram file path")
    parser.add_argument("-o", "--output_file", help="output_file for figure.",
                        default=None)
    parser.add_argument("-t", "--title", help="title of the plot",
                        default=None)
    parser.add_argument("-f", "--bandpass", nargs=2, type=float, default=None,
                        metavar=("min_freq", "max_freq"),
                        help="bandpass filter")

    # Reading
    args = parser.parse_args()
    seismogram_file = Path(args.seismogram)
    tr = read_ascii_trace(seismogram_file)

    # Filtering
    if args.bandpass:
        fmin, fmax = args.bandpass
        tr.filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)

    # Plotting
    fig, ax = plot_trace(tr)
    ax.set_title(args.title or seismogram_file.stem)
    fig.tight_layout()
    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()
