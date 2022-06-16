#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Puts a slow velocity zone in a SPECFEM2D ascii model.

Ridvan Orsvuran, 2022"""

import numpy as np


model = np.loadtxt("./DATA/proc000000_rho_vp_vs.dat")
rows, columns = model.shape

XMIN, XMAX = 1000, 2000
ZMIN, ZMAX = 1000, 2000

for i in range(rows):
    # X Z rho vp vs
    x, z, rho, vp, vs = model[i, :]
    if XMIN <= x <= XMAX and ZMIN <= z <= ZMAX:
        # half Vp and Vs
        model[i, 3] = vp / 2.0
        model[i, 4] = vp / 2.0

np.savetxt("./DATA/proc000000_rho_vp_vs.dat", model)
