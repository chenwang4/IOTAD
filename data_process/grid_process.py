#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:05:23 2022

@author: cww3
"""

import random
import numpy as np
import pandas as pd
import os.path
import os
from pykalman import KalmanFilter

def neighbouring(i, k):
    # i: (x, y) int tuple.  k: (x, y) int tuple.
    return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

def point_to_int(p,grid_size):
    return p[0] + p[1]*grid_size

def gps_to_gw(traj_gps,cols,rows,grid_size):
    x = np.searchsorted(cols,traj_gps[:,1])
    y = np.searchsorted(rows,traj_gps[:,0])
    x = x[::10]
    y = y[::10]
    traj_mdp = []
    for i in range(1,len(x)):
        prev_state = point_to_int((x[i-1],y[i-1]),grid_size)
        cur_state = point_to_int((x[i],y[i]),grid_size)
        if prev_state != cur_state:
        #get interpolation actions and states
            part_traj = get_action_interp(cur_state, prev_state,grid_size)
            traj_mdp = traj_mdp + part_traj
        else:
            action = 0
            part_traj = (prev_state, action)
            traj_mdp.append(part_traj)
    return traj_mdp

def get_action_interp(cur_state, prev_state,grid_size):
    # prevs = []
    # acts = []
    # curs = []
    trajectory = []
    cur_row = cur_state // grid_size
    cur_col = cur_state % grid_size
    prev_row = prev_state // grid_size
    prev_col = prev_state % grid_size
    rel_row = cur_row - prev_row
    rel_col = cur_col - prev_col
    while not (rel_row == 0 and rel_col == 0):
        action = 0
        if rel_row > 0 and rel_col > 0:
            action = random.choice([1,2])
        elif rel_row > 0 and rel_col == 0:
            action = 1 #go up
        elif rel_row > 0 and rel_col < 0:
            action = random.choice([1,4])
        elif rel_row == 0 and rel_col > 0:
            action = 2 #go right
        elif rel_row == 0 and rel_col < 0:
            action = 4 #go left
        elif rel_row < 0 and rel_col > 0:
            action = random.choice([3,2])
        elif rel_row < 0 and rel_col == 0:
            action = 3 #go down
        elif rel_row < 0 and rel_col < 0:
            action = random.choice([3,4])
        row_diff = -np.sign(rel_row) if (action == 1 or action == 3) else 0
        col_diff = -np.sign(rel_col) if (action == 2 or action == 4) else 0
        rel_row += row_diff
        rel_col += col_diff
        temp_row = prev_row - row_diff
        temp_col = prev_col - col_diff
        temp_state = temp_row * grid_size + temp_col
        prev_state = prev_row * grid_size + prev_col

        # records an interpolated state-action-state transition
        trajectory.append((prev_state, action))
        # prevs.append(prev_state)
        # acts.append(action)
        # curs.append(temp_state)
        prev_row = temp_row
        prev_col = temp_col
    
    return trajectory

def data_process(trajs, grid_size,bottom_left, bottom_right, top_left): #bottom_left, bottom_right, top_left
    trajs_smoothed = []
    for i in range((len(trajs))):
        measurements = trajs[i]
        kf = KalmanFilter(initial_state_mean = measurements[0], \
               n_dim_obs=2, n_dim_state=2)
        measurements = np.stack(measurements).astype(None)
        kf = kf.em(measurements)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
        trajs_smoothed.append(smoothed_state_means)
    #trajs_smoothed.remove(trajs_smoothed[46])
    # flat_lati = [item for sublist in trajs_smoothed for item in sublist[:,0]]
    # flat_longi = [item for sublist in trajs_smoothed for item in sublist[:,1]]
    # bottom_left = (min(flat_lati)-0.001,min(flat_longi)-0.001)
    # bottom_right = (min(flat_lati)-0.001,max(flat_longi)+0.001)
    # top_left = (max(flat_lati)+0.001,min(flat_longi)-0.001)
    # top_right = (max(flat_lati)+0.001,max(flat_longi)+0.001)
    cols = np.linspace(bottom_left[1],bottom_right[1],num=grid_size-1)
    rows = np.linspace(bottom_left[0],top_left[0],num=grid_size-1)
    trajs_gw = []
    for i in range(len(trajs_smoothed)):
        traj_single = gps_to_gw(trajs_smoothed[i],cols,rows,grid_size)
        trajs_gw.append(traj_single)
    return trajs_gw,trajs_smoothed

def sources_to_gw(traj_gw):
    s_gw = []
    for traj in traj_gw:
        s_gw.append(traj[0][0])
    return s_gw
    
    
    
    
    
    
    
    
    