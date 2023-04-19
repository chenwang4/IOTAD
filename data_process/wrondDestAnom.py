#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:55:52 2022

@author: cww3

generate wrong-destination anomaly sets for 'Chengdu didi dataset'
"""

import random
import numpy as np
#import mdp 
from shapely.geometry import Point, Polygon
import grid_process as process


class data_generation:
    def __init__(self, other_trajs, boundary, grid_size, win_len, step_size, pred_win,traj_allcluster):
        self.other_trajs = other_trajs
        self.grid_size = grid_size
        self.map_poly = Polygon(boundary)
        self.col_diff = (boundary[1][1] - boundary[0][1])/grid_size
        self.row_diff = (boundary[2][0] - boundary[0][0])/grid_size
        self.win_len = win_len
        self.step_size = step_size
        self.pred_win = pred_win
        self.traj_allcluster = traj_allcluster
    def dest_int_to_point(self,trajs):
        dest = []
        for t in trajs:
            dest.append(list(t[-1]))
        #dest = list(set(dest))
        return dest
    def select_anomaly_wd(self):
        traj_all = self.other_trajs
        dest_anomaly = self.dest_int_to_point(traj_all)
        polys = self.normal_destination_region()
        i = 0
        d_index = []
        for d in dest_anomaly:
            p = Point(d[0], d[1])
            flag = 1
            for poly in polys:
                if p.within(poly):
                    flag = 0
                    break
            if flag == 1 and p.within(self.map_poly):
                d_index.append(i)
            i += 1

        traj_anomaly = [traj_all[d] for d in d_index]
        traj_anomaly_ = []
        for i in range(len(traj_anomaly)):  
            traj = traj_anomaly[i][::15]
            traj_len = len(traj)
            num_sec = int((traj_len - self.pred_win - self.win_len)/self.step_size + 1)
            if num_sec > 3:
                traj_anomaly_.append(traj_anomaly[i])
        return traj_anomaly_
    def normal_destination_region(self):
        # userdata = '/Users/cww3/Desktop/GeolifeTrajectories /Data/'+ self.target_user + '/Trajectory/'
        # traj_allcluster, sources_allcluster = DC.dest_cluster(userdata)
        trajs_normal = []
        for i in self.traj_allcluster:
            trajs_normal = np.append(trajs_normal, i, axis = 0)
        dest_normal = self.dest_int_to_point(trajs_normal)
        polys = []
        for d in dest_normal:
            coords = [(d[0]-8*self.row_diff, d[1]), (d[0], d[1]-8*self.col_diff), (d[0]+8*self.row_diff, d[1]), (d[0], d[1]+8*self.col_diff)]
            poly = Polygon(coords)        
            polys.append(poly)
        return polys
    
def wd_anomaly(trajs_allcluster):
    other_trajs = []
    for i in range(8,60):
        other_trajs = np.append(other_trajs, trajs_allcluster[i])
    #boundary = [(39.872118, 116.332948),(39.872118, 116.479633),(40.000042, 116.479633), (40.000042, 116.332948)]
    boundary = [(30.65294, 104.04216),(30.65294, 104.12952),(30.72732,104.12952),(30.72732, 104.04216)]
    grid_size = 50
    win_len = 20
    step_size = 2
    pred_win = 10
    traj_allcluster = trajs_allcluster[0:4]
    DG_wd = data_generation(other_trajs, boundary, grid_size, win_len, step_size, pred_win,traj_allcluster)
    anomaly_wd = DG_wd.select_anomaly_wd()
    traj_anomaly_s1 = random.sample(anomaly_wd, 100)
    traj_anomaly_s2 = random.sample(anomaly_wd, 100)
    traj_anomaly_s3 = random.sample(anomaly_wd, 100)
    traj_anomaly_s4 = random.sample(anomaly_wd, 100)
    traj_anomaly_s5 = random.sample(anomaly_wd, 100) 
    bottom_left = (30.65294, 104.04216)
    bottom_right = (30.65294, 104.12952)
    top_left = (30.72732, 104.04216)
    traj_anomaly_s1_gw,_ = process.data_process(traj_anomaly_s1, grid_size, bottom_left, bottom_right, top_left)
    traj_anomaly_s2_gw,_ = process.data_process(traj_anomaly_s2, grid_size, bottom_left, bottom_right, top_left)
    traj_anomaly_s3_gw,_ = process.data_process(traj_anomaly_s3, grid_size, bottom_left, bottom_right, top_left)
    traj_anomaly_s4_gw,_ = process.data_process(traj_anomaly_s4, grid_size, bottom_left, bottom_right, top_left)
    traj_anomaly_s5_gw,_ = process.data_process(traj_anomaly_s5, grid_size, bottom_left, bottom_right, top_left)
    anomaly_wd_g5 = [traj_anomaly_s1_gw,traj_anomaly_s2_gw,traj_anomaly_s3_gw,traj_anomaly_s4_gw,traj_anomaly_s5_gw]
    traj_wd_ext = []
    for g in anomaly_wd_g5:
        ext_group = []
        for t in g:
            traj = t.copy()
            traj.extend([traj[-1]] * 15)
            ext_group.append(traj)
        traj_wd_ext.append(ext_group)
    return traj_wd_ext
    


