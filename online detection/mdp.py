#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:20:55 2022

@author: cww3
"""
import numpy as np
class mdp(object):
    def __init__(self, grid_size, determinism, discount):
        self.actions = ((0,0),(0, 1), (1, 0), (0,-1), (-1,0))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.n = grid_size
        self.determinism = determinism
        self.discount = discount
        self.P_transition = self.transition()
    
    def transition(self):
        transition_probability = np.zeros((self.n_states, self.n_actions, self.n_states))
        sa_s = np.zeros((self.n**2,5,5), int)
        sa_p = np.zeros((self.n**2,5,5))

        for y in range(self.n):
            for x in range(self.n):

                s = y*self.n + x + 1
                successors = np.zeros((1,1,5))
                successors[0,0,0] = s - 1 
                successors[0,0,1] = (min(self.n,y+2)-1)*self.n + x + 1 - 1
                successors[0,0,2] = y*self.n + min(self.n,x+2) - 1
                successors[0,0,3] = (max(1,y)-1)*self.n+x+1 - 1
                successors[0,0,4] = y*self.n+max(1,x) - 1
                sa_s[s-1,:,:] = np.tile(successors, (1, 5, 1))
                sa_p[s-1,:,:] = np.reshape(
                    np.eye(5)*self.determinism + (np.ones(5) - np.eye(5))*((1 - self.determinism)/4), 
                    (1, 5, 5)
                )
        for k in range(self.n_states):
            for a in range(self.n_actions):
                for j in range(self.n_actions):
                    next_s = sa_s[k][a][j]
                    transition_probability[k, a, next_s] += sa_p[k][a][j]
        return transition_probability
    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.n)
            x, y = i % self.n, i // self.n
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.n, i // self.n
            for b in range(self.n):
                for a in range(self.n):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f   
    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)    
    
    
    
    
    
    