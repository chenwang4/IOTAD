#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:14:43 2022

@author: cww3
"""
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import MeanShift
import grid_process as process 

def get_distance(s,d):
    R = 6373.0
    lat1 = radians(s[0])
    lon1 = radians(s[1])
    lat2 = radians(d[0])
    lon2 = radians(d[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
def data_generation():
    # each order id can be considered as a trajectory
    userdata = '/Users/cww3/Desktop/icdm_irl_anomaly_detection/code/DiDi/gps_01_10/gps_20161101'
    #userdata = '/Users/cww3/Desktop/DiDi/gps_01_10/gps_20161101'
    #userdata = './/..//gps_20161101'
    names = ['driver_id', 'order_id', 'time', 'longi', 'lati']
    df = pd.read_table(userdata, header = None, names=names, sep=',')
    lat_lon_data = np.c_[df['lati'].values, df['longi'].values, df['order_id'].values]
    trajs = []
    s_ind = 0
    d_ind = 1  
    for i in range(len(lat_lon_data)-1):
        if lat_lon_data[i][2] == lat_lon_data[i+1][2]:
            d_ind += 1
        else:
            traj = lat_lon_data[s_ind:d_ind]
            trajs.append(traj)
            s_ind = i + 1
            d_ind += 1   
    # extract longer trajectories
    trajs_longer = []
    dests = []
    dists = []
    for t in trajs:
        dist = get_distance(t[0][0:2],t[-1][0:2])    
        if dist > 3 and dist < 9:
            trajs_longer.append(t[:,0:2])
            dists.append(dist)
            dests.append(list(t[-1][0:2]))
    # clustering based on destinations
    clustering = MeanShift(bandwidth=0.004).fit(dests)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    trajs_longer = np.array(trajs_longer)
    trajs_allcluster = []
    for label in range(n_clusters_):
        trajs_ind = np.where(labels==label)[0]   
        if len(trajs_ind) > 100:
            traj_cluster = trajs_longer[trajs_ind]
            trajs_allcluster.append(traj_cluster)
    bottom_left = (30.65294, 104.04216)
    bottom_right = (30.65294, 104.12952)
    top_left = (30.72732, 104.04216)
    grid_size = 50
    n_clusters = 3
    traj_cluster_gw = []
    for i in range(n_clusters):
        group = trajs_allcluster[i]
        trajs_gw, traj_smooth = process.data_process(group, grid_size,bottom_left, bottom_right, top_left)
        trajs_gw_ = list(filter(None,trajs_gw))
        traj_cluster_gw.append(trajs_gw_)
    return trajs_allcluster, traj_cluster_gw


# #%%
# group_a = trajs_allcluster[0][0:1000]
# group_b = trajs_allcluster[1][0:1000]
# group_c = trajs_allcluster[2][0:1000]
# group_d = trajs_allcluster[3][0:1000]
# group = np.concatenate((group_a, group_b,group_c),axis=0)
# bottom_left = (30.65294, 104.04216)
# bottom_right = (30.65294, 104.12952)
# top_left = (30.72732, 104.04216)
# grid_size = 50
# discount = 0.95
# determinism = 0.9
# gw = mdp.mdp(grid_size, determinism, discount)
# feat_map = gw.feature_matrix()
# trajs_gw, traj_smooth = process.data_process(group_a, grid_size,bottom_left, bottom_right, top_left)
# #%%
# learning_rate = 0.1
# epochs = 100
# rewards = maxent.deep_maxent_irl(feat_map, gw.P_transition, discount, trajs_gw, learning_rate, epochs)
# np.save('r1.npy',rewards)

# group_c = trajs_allcluster[2]

# trajs_gw, traj_smooth = process.data_process(group_c, grid_size,bottom_left, bottom_right, top_left)
# [] in trajs_gw
# trajs_gw_ = list(filter(None,trajs_gw))
