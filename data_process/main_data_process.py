#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:54:11 2022

@author: cww3
"""

import numpy as np
import data_process
from sklearn.model_selection import KFold
import value_iteration
import mdp
import random
import wrondDestAnom as WDA
import synthetic_anomaly 
grid_size = 50
discount = 0.95
determinism = 0.9
n_cluster = 3
gw = mdp.mdp(grid_size, determinism, discount)

def get_data():
    """
    for Chengdu dataset, transfer gps trajectories to grid trajectories 

    Returns clustered grid world trajectories, training and testing data for each destination-based cluster
   
    """
    trajs_allcluster, traj_allcluster_gw = data_process.data_generation()
    kf = KFold(n_splits=3,random_state=None, shuffle=True)
    #kf.get_n_splits(traj_normal)
    X_train_cluster = []
    X_test_cluster = []
    for i in range(len(traj_allcluster_gw)):
        for train_index, test_index in kf.split(traj_allcluster_gw[i]):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = np.array(traj_allcluster_gw[i])[train_index], np.array(traj_allcluster_gw[i])[test_index]
        X_train_cluster.append(x_train)
        X_test_cluster.append(x_test)
    return trajs_allcluster, traj_allcluster_gw, X_train_cluster, X_test_cluster
def remove_action(traj_group):
    new_group = []
    for group in traj_group:
        s_group = []
        for traj in group:
            s_traj = []
            for s,a in traj:
                s_traj.append(s)
            s_group.append(s_traj)
        new_group.append(s_group)
    return new_group

trajs_allcluster, traj_allcluster_gw, X_train_cluster, X_test_cluster = get_data()
train_cluster = remove_action(X_train_cluster)
test_cluster = remove_action(X_test_cluster)
r_1 = np.load('../r1.npy')
r_1 = r_1.reshape(-1)
p_1, v_1 = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.P_transition, r_1, discount,
                threshold=1e-2, v=None, stochastic=True)
r_2 = np.load('../r2.npy')
r_2 = r_2.reshape(-1)
p_2, v_2 = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.P_transition, r_2, discount,
                threshold=1e-2, v=None, stochastic=True)
r_3 = np.load('../r3.npy')
r_3 = r_3.reshape(-1)
p_3, v_3 = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.P_transition, r_3, discount,
                threshold=1e-2, v=None, stochastic=True)
v = [v_1, v_2, v_3]
wind = 0
gw_anomaly = synthetic_anomaly.Gridworld_anomaly(grid_size, wind, discount)
#random walk
repeat_times=[1,1,1,2,2]
sources_1 = []
for t in traj_allcluster_gw[0]:
    sources_1.append(t[0][0])
sources_2 = []
for t in traj_allcluster_gw[1]:
    if t != []:
        sources_2.append(t[0][0])
sources_3 = []
for t in traj_allcluster_gw[2]:
    if t != []:
        sources_3.append(t[0][0])
sources = [sources_1,sources_2,sources_3]
r = [r_1,r_2,r_3]
rw_c1 = gw_anomaly.generate_ramdomwalk(200, sources[0], r[0], 0.4,repeat_times)
rw_c2 = gw_anomaly.generate_ramdomwalk(200, sources[1], r[1], 0.4,repeat_times)
rw_c3 = gw_anomaly.generate_ramdomwalk(200, sources[2], r[2], 0.4,repeat_times)
rw_sum = rw_c1 + rw_c2 + rw_c3
trajs_rw = []
for i in range(5):
    trajs_rw.append(random.sample(rw_sum, 100))
anomaly_rw_g5 = remove_action(trajs_rw)
#detour
traj_normal = []
for t in traj_allcluster_gw[0:n_cluster]:
    for tt in t:
        traj_normal.append(tt)
anomaly_20_sum = []
for i in range(5):
    anomaly_20, traj_n_20 = gw_anomaly.generate_detour(100, traj_normal, 20,repeat_times,alpha=0.6)
    anomaly_20_sum.append(anomaly_20)
anomaly_d20_g5 = remove_action(anomaly_20_sum)

#wrong destination
traj_wd_ext = WDA.wd_anomaly(trajs_allcluster)
anomaly_wd = remove_action(traj_wd_ext)


data_dir = './anomaly'
data_name = "Chengdu_randomwalk"
fout = open("{}/processed_{}.csv".format(data_dir, data_name), 'w')
for g in anomaly_rw_g5:
    for traj in g:
        fout.write("[")
        for s in traj[:-1]:
            fout.write("%s, " % str(s))
        fout.write("%s]\n" % str(traj[-1]))
fout.close()

data_dir = './anomaly'
data_name = "Chengdu_wrongDestination"
fout = open("{}/processed_{}.csv".format(data_dir, data_name), 'w')
for g in anomaly_wd:
    for traj in g:
        fout.write("[")
        for s in traj[:-1]:
            fout.write("%s, " % str(s))
        fout.write("%s]\n" % str(traj[-1]))
fout.close()

data_dir = './anomaly'
data_name = "Chengdu_detour"
fout = open("{}/processed_{}.csv".format(data_dir, data_name), 'w')
for g in anomaly_d20_g5:
    for traj in g:
        fout.write("[")
        for s in traj[:-1]:
            fout.write("%s, " % str(s))
        fout.write("%s]\n" % str(traj[-1]))
fout.close()