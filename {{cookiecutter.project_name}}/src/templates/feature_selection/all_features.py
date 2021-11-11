#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
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
selected = [True for _ in featnames]
scores = [0 for _ in featnames]

u.save_scores_npz(featnames, selected, scores, {})
