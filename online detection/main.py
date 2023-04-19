#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:44:04 2022

@author: cww3

Online detetcion results
"""

import numpy as np
import random
import value_iteration
import mdp
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.metrics import precision_recall_curve, auc
random.seed(10)
grid_size = 50
discount = 0.95
determinism = 0.9
gw = mdp.mdp(grid_size, determinism, discount)
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
data_dir = './anomaly'
data_name = "Chengdu_wrongDestination"
def read_traj(data_dir, data_name):
    f = open("{}/processed_{}.csv".format(data_dir, data_name), 'r').readlines()
    trajs = []
    for eachline in f:
        traj = eval(eachline)
        trajs.append(traj)
    return trajs
anomaly_wd = read_traj('./anomaly', "Chengdu_wrongDestination")
anomaly_detour = read_traj('./anomaly', "Chengdu_detour")
anomaly_rw = read_traj('./anomaly', "Chengdu_randomwalk")
anomalies = anomaly_wd + anomaly_detour + anomaly_rw

train_normal = read_traj('./data',"Chengdu_train") #2113+2108+1802
X_train_cluster = [train_normal[0:2113],train_normal[2113:4221],train_normal[4221:]]
val_normal = read_traj('./data',"Chengdu_val") #1056+1054+901
X_test_cluster = [val_normal[0:1056],val_normal[1056:2110],val_normal[2110:]]
win_len = 20
step_size = 2
pred_win = 10
num_feat = 1
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
def feature_traj(trajectories, v_function):
    x_feat = []
    y_feat = []
    sections = []
    for traj in trajectories:
        traj_len = len(traj)
        f_temp = []
        for s in traj:
            v = v_function[int(s)]
            f_temp.append([v]) #s % grid_size, s // grid_size,
        #T = 20
        win_len = 20
        step_size = 2
        pred_win = 10
        num_section = int((traj_len - pred_win - win_len)/step_size + 1)
        if num_section < 1:
            continue
        f = []
        y = []
        for k in range(num_section):
            w = np.array(f_temp[step_size*k:win_len + step_size*k]).reshape(-1)
            w2 = np.array(f_temp[step_size*k + win_len: pred_win + step_size*(k) + win_len]).reshape(-1)
            y.append(w2)
            f.append(w)
        x_feat.append(f)
        y_feat.append(y)
        sections.append(num_section)
    return x_feat, y_feat, sections

def data_reshape(trajs, v):
    x_, y_, sec_ = feature_traj(trajs, v)
    x_train = []
    for x in x_:
        for xx in x:
            x_train.append(xx)
    y_train = []
    for x in y_:
        for xx in x:
            y_train.append(xx)
    #x_train = np.array(x_train).reshape(-1,win_len,num_feat).astype(np.float32)
    #y_train = np.array(y_train).reshape(-1,pred_win,num_feat).astype(np.float32)
    return x_train, y_train, sec_
def model_train(x_train_, y_train_):
    pred_win = 10
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train_.shape[1], x_train_.shape[2])))
    model.add(Dropout(rate=0.1))
    model.add(RepeatVector(pred_win))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.1))
    model.add(TimeDistributed(Dense(x_train_.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    model.fit(x_train_, y_train_, epochs=100, batch_size=32, validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')], shuffle=True)
    return model

def detect(traj_gw,V,model,n_cluster):
    test_losses = []
    for i in range(n_cluster):
        x_, y_, sec_ = data_reshape(traj_gw, V[i])
        x_ = np.array(x_).reshape(-1,win_len,num_feat).astype(np.float32)
        y_ = np.array(y_).reshape(-1,pred_win,num_feat).astype(np.float32)
        X_pred = model.predict(x_, verbose=0)
        test_mae_loss = np.mean(np.abs(X_pred - y_), axis=1)
        test_losses.append(test_mae_loss)
    losses_binary = []
    for i in range(len(test_losses[0])):
        min_loss = min(test_losses[0][i],test_losses[1][i],test_losses[2][i])
        losses_binary.append(min_loss)
    len_accu = 0
    losses_final = []
    for i in sec_:
        max_loss = max(losses_binary[len_accu:len_accu+i]) #for one trajectory
        losses_final.append(max_loss)
        len_accu = len_accu + i
    return losses_final

def get_auc(test_cluster, anomalies, v, model):
    n_cluster = 3
    aucs = []
    for g in range(5):
        x_test = []
        for i in range(n_cluster):
            x_test = x_test + random.sample(test_cluster[i],300)
        losses_normal = detect(x_test, v, model, n_cluster)
        losses_anomalies = detect(random.sample(anomalies, 150), v, model, n_cluster)
        y_true = np.concatenate((np.zeros(len(losses_normal)),np.ones(len(losses_anomalies))))
        y_loss = np.array(losses_normal + losses_anomalies)
        p,r,t = precision_recall_curve(y_true,y_loss)
        aucs.append(auc(r,p))
    return np.mean(aucs), np.std(aucs)

def train_test(X_train, X_test, train_size, anomalies, v):
    models = []
    n_cluster = 3
    for k in range(5):
        x_train_ = []
        y_train_ = []
        for i in range(n_cluster):
            traj_samp = random.sample(X_train[i], train_size)
            x_1, y_1, sec_1 = data_reshape(traj_samp, v[i])
            x_train_+=x_1
            y_train_+=y_1
        x_train_ = np.array(x_train_).reshape(-1,win_len,num_feat).astype(np.float32)
        y_train_ = np.array(y_train_).reshape(-1,pred_win,num_feat).astype(np.float32)
        m = model_train(x_train_, y_train_)
        models.append(m)
    aucs = []
    auc_stds = []
    for each_model in models:
        auc_mean, auc_std = get_auc(X_test, anomalies, v, each_model)
        aucs.append(auc_mean)
        auc_stds.append(auc_std)      
    return models[np.argmax(aucs)], max(aucs), auc_stds[np.argmax(aucs)]

train_size = 100
values = [v_1, v_2, v_3]
best_model, auc_mean, auc_std = train_test(X_train_cluster, X_test_cluster, train_size, anomalies, values)
