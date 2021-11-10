#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector
Output files:
  - selected.npz: contains the featnames of all the features
"""
import utils as u

u.set_random_state()

# Read data
############################
_, _, featnames = u.read_data("${TRAIN_NPZ}")

# Save selected features
############################
u.save_selected_npz(featnames)
