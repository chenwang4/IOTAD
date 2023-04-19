#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:28:53 2022

@author: cww3

synthetic anomaly generation
"""

import numpy as np
import numpy.random as rn
import value_iteration
import random
class Gridworld_anomaly(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((0,0),(0, 1), (1, 0), (0,-1), (-1,0))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        if feature_map == 'trajectory':
            f = np.zeros(5) 
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

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1
    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.
        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk) and (xj, yj)!=(0,0):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions
        


        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 3*self.wind/self.n_actions
            else:
                if (xj, yj) == (0, 0):
                  return 1 - self.wind + 3*self.wind/self.n_actions
                else:
                # We can blow off the grid in either direction only by wind.
                  return 3*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}) and (xj, yj)==(0,0):
                # Not an edge.
                return 1 - self.wind + self.wind/self.n_actions
            
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}) and (xj, yj)!=(0,0):
                # Not an edge.
                return self.wind/self.n_actions

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                if (xj, yj) == (0, 0):
                  return 1 - self.wind + 2*self.wind/self.n_actions
                else:
                # We can blow off the grid only by wind.
                  return 2*self.wind/self.n_actions



    def reward(self, state_int, dest1):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == dest1:
            return 1

        return 0
#        if dest2 is not None and state_int == self.n_states - dest2:
#            return 0.35
#        x, y = self.int_to_point(state_int)
#        if x < 2 or y < 2 or x > 7 or y > 7:
#            return 0.1
#        else:
#            return -1

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1
    def generate_ramdomwalk(self, n_trajectories, sources, r_given, rw_degree, repeat_times):
        trajectories = []
        policy_gt, v_gt = value_iteration.find_policy(self.n_states, self.n_actions,
                                self.transition_probability, r_given, self.discount, stochastic=True)
        for _ in range(n_trajectories): 
            state_int = random.choice(sources)
            sx, sy = self.int_to_point(state_int)
            trajectory = []
            while r_given[state_int] < 0.96:
                action_prob = policy_gt[self.point_to_int((sx, sy))]
                if rn.random() < rw_degree:
                    action = self.actions[rn.randint(0, 5)]
                else:
                    action = self.actions[np.argmax(action_prob)]
                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                rep = random.choice(repeat_times)
                traj_part = [(state_int, action_int)]*rep
                trajectory = trajectory + traj_part
                #next_state_int = self.point_to_int((next_sx, next_sy))

                #trajectory.append((state_int, action_int))

                sx = next_sx
                sy = next_sy
            trajectories.append(trajectory)
        return trajectories

            
            
    # def generate_detour(self, n_trajectories, sources, destinations, deviation, repeat_times):
    #     """
    #     Generate n_trajectories trajectories with length trajectory_length,
    #     following the given policy.

    #     n_trajectories: Number of trajectories. int.
    #     trajectory_length: Length of an episode. int.
    #     policy: Map from state integers to action integers.
    #     random_start: Whether to start randomly (default False). bool.
    #     -> [[(state int, action int, reward float)]]
    #     """

    #     trajectories = []
    #     for _ in range(n_trajectories):
    #         sx, sy = self.int_to_point(random.choice(sources))
    #         dx, dy = self.int_to_point(random.choice(destinations))
    #         dest = self.point_to_int((dx, dy))
    #         midx, midy = self.select_midpoint(deviation, (dx,dy), (sx,sy))
    #         mp = self.point_to_int((midx, midy))
    #         #generate policy 1 using mid point as destination
    #         r_mid = np.array([self.reward(s,mp) for s in range(self.n_states)])
            
    #         policy1, v1 = value_iteration.find_policy(self.n_states, self.n_actions,
    #                                 self.transition_probability, r_mid, discount, stochastic=True)
    #         #generate policy 2 using final (true) destination as the destination
    #         r_op = np.array([self.reward(s,dest) for s in range(self.n_states)])
    #         policy2, v2 = value_iteration.find_policy(self.n_states, self.n_actions,
    #                                 self.transition_probability, r_op, discount, stochastic=True)
          
    #         trajectory = []
    #         c = 0
    #         flag = 0
    #         while c == 0:
    #             # Follow the given policy.
    #             if flag == 0:
    #                 action_prob = policy1[self.point_to_int((sx, sy))]
    #             else:
    #                 action_prob = policy2[self.point_to_int((sx, sy))]
    #             if all(x==action_prob[0] for x in action_prob):
    #                 action = self.actions[rn.randint(0, self.n_actions)]
    #             else:
    #                 action = self.actions[np.argmax(action_prob)]
        
    #             if (0 <= sx + action[0] < self.grid_size and
    #                     0 <= sy + action[1] < self.grid_size):
    #                 next_sx = sx + action[0]
    #                 next_sy = sy + action[1]
    #             else:
    #                 next_sx = sx
    #                 next_sy = sy

    #             state_int = self.point_to_int((sx, sy))
    #             action_int = self.actions.index(action)
    #             next_state_int = self.point_to_int((next_sx, next_sy))
    #             if next_state_int == mp:
    #                 flag = 1
    #             rep = random.choice(repeat_times)
    #             traj_part = [(state_int, action_int)]*rep
    #             trajectory = trajectory + traj_part

    #             sx = next_sx
    #             sy = next_sy
    #             if state_int == dest:
    #                 c += 1

    #         trajectories.append(trajectory)

    #     return trajectories
    
    def generate_detour(self, n_trajectories, trajs_n_gw, deviation, repeat_times, alpha):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        trajs_norm = []
        for _ in range(n_trajectories):
            # select a normal traj
            traj_norm_r = random.choice(trajs_n_gw)
            # remove the repeated values in this normal traj
            traj_norm = self.remove_adjacent(traj_norm_r)
            # select source and destination based on alpha 
            # alpha is the proportion of anomalous traj
            
            thres = int(len(traj_norm) * (1 - alpha))
            ind_source = random.choice(range(0,thres))
            ind_dest = ind_source+len(traj_norm)-thres
            s_int = traj_norm[ind_source][0]
            dest_int = traj_norm[ind_dest][0]
            sx, sy = self.int_to_point(s_int)
            dx, dy = self.int_to_point(dest_int)
            # add detour to this partial trajectory
            midx, midy = self.select_midpoint(deviation, (dx,dy), (sx,sy))
            mp = self.point_to_int((midx, midy))
            #generate policy 1 using mid point as destination
            r_mid = np.array([self.reward(s,mp) for s in range(self.n_states)])
            
            policy1, v1 = value_iteration.find_policy(self.n_states, self.n_actions,
                                    self.transition_probability, r_mid, self.discount, stochastic=True)
            #generate policy 2 using final (true) destination as the destination
            r_op = np.array([self.reward(s,dest_int) for s in range(self.n_states)])
            policy2, v2 = value_iteration.find_policy(self.n_states, self.n_actions,
                                    self.transition_probability, r_op, self.discount, stochastic=True)
          
            trajectory = traj_norm[0:ind_source]
            
            c = 0
            flag = 0
            while c == 0:
                # Follow the given policy.
                if flag == 0:
                    action_prob = policy1[self.point_to_int((sx, sy))]
                else:
                    action_prob = policy2[self.point_to_int((sx, sy))]
                if all(x==action_prob[0] for x in action_prob):
                    action = self.actions[rn.randint(0, self.n_actions)]
                else:
                    action = self.actions[np.argmax(action_prob)]
        
                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                if next_state_int == mp:
                    flag = 1
                rep = random.choice(repeat_times)
                traj_part = [(state_int, action_int)]*rep
                trajectory = trajectory + traj_part

                sx = next_sx
                sy = next_sy
                if state_int == dest_int:
                    c += 1
            trajectory = trajectory + traj_norm[ind_dest:]
            trajectories.append(trajectory)
            trajs_norm.append(traj_norm)
        return trajectories, trajs_norm
    
    
    def deviation_comp(self, i, dest, start):
        # if i[0] > dest[0] and i[1] <= dest[1]:
        #     d = (i[0] - dest[0]) * 2
        # elif i[0] <= dest[0] and i[1] > dest[1]:
        #     d = (i[1] - dest[1]) * 2
        # elif i[0] > dest[0] and i[1] > dest[1]:
        #     d = (i[0] - dest[0]) * 2 + (i[1] - dest[1]) * 2
        # else:
        #     d = 0
        if (i[0] < dest[0] and i[0] > start[0]) or (i[0] > dest[0] and i[0] < start[0]):
            dx = 0
        else:
            dx = 2 * min(abs(i[0]-dest[0]), abs(i[0]-start[0]))
        if (i[1] < dest[1] and i[1] > start[1]) or (i[1] > dest[1] and i[1] < start[1]):
            dy = 0
        else:
            dy = 2 * min(abs(i[1]-dest[1]), abs(i[1]-start[1]))
        d = dx + dy
        return d

    def select_midpoint(self, deviation, dest, start):
        mid_point = []
        for i in range(self.n_states):
            s = self.int_to_point(i)
            dev = self.deviation_comp(s, dest, start)
            if dev == deviation:
                mid_point.append(s)
        return random.choice(mid_point)
    def remove_adjacent(self, num):
      i = 1
      nums = num.copy()
      while i < len(nums):    
        if nums[i] == nums[i-1]:
          nums.pop(i)
          i -= 1  
        i += 1
      return nums
    def generate_normal(self, n_trajectories, policy, dest, repeat_times,
                                  random_start=True):
      """
      Generate n_trajectories trajectories with length trajectory_length,
      following the given policy.

      n_trajectories: Number of trajectories. int.
      trajectory_length: Length of an episode. int.
      policy: Map from state integers to action integers.
      random_start: Whether to start randomly (default False). bool.
      -> [[(state int, action int, reward float)]]
      """

      trajectories = []
      for _ in range(n_trajectories):
          if random_start:
              sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
          else:
              sx, sy = 0, 0

          trajectory = []
          c = 0
          while c <=2:
              if rn.random() < self.wind:
                  action = self.actions[rn.randint(0, 4)]
              else:
                  # Follow the given policy.
                  action = self.actions[policy[self.point_to_int((sx, sy))]]

              if (0 <= sx + action[0] < self.grid_size and
                      0 <= sy + action[1] < self.grid_size):
                  next_sx = sx + action[0]
                  next_sy = sy + action[1]
              else:
                  next_sx = sx
                  next_sy = sy

              state_int = self.point_to_int((sx, sy))
              action_int = self.actions.index(action)
              rep = random.choice(repeat_times)
              traj_part = [(state_int, action_int)]*rep
              #next_state_int = self.point_to_int((next_sx, next_sy))
              #reward = self.reward(next_state_int,dest)
              trajectory = trajectory + traj_part

              sx = next_sx
              sy = next_sy
              if state_int == dest:
                  c += 1
          trajectories.append(trajectory)

      return trajectories



# grid_size = 50
# discount = 0.95
# wind = 0
# gw = Gridworld_anomaly(grid_size, wind, discount)
# #z,m=gw.select_midpoint(2, (29,37),(28,47))


# #%%
# # calculate repeat times in normal trajs, most frequent and proportion
# repeat_times=[1,1,2,3,3,4]
# anomaly_16, traj_n = gw.generate_detour(4,trajs_gw, 16,repeat_times,alpha=0.3)
# import matplotlib.pyplot as plt
# rewards = np.load('/Users/cww3/Desktop/reward/user124/r3.npy')
# #%%
# n = 3
# plt.pcolor(rewards.reshape((grid_size, grid_size)))
# plt.colorbar()
# last_position = gw.int_to_point(traj_n[n][0][0])
# for t in traj_n[n]:
#     state = t[0]
#     current_state = state
#     current_position = gw.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw.int_to_point(last_state)
# plt.show()
# import matplotlib.pyplot as plt
# plt.pcolor(r_1.reshape((grid_size, grid_size)))
# plt.colorbar()
# n=6
# last_position = gw_anomaly.int_to_point(anomaly_12[n][0][0])
# for t in anomaly_12[n]:
#     state = t[0]
#     current_state = state
#     current_position = gw_anomaly.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw_anomaly.int_to_point(last_state)
# plt.show()



# plt.pcolor(r_1.reshape((grid_size, grid_size)))
# plt.colorbar()
# last_position = gw_anomaly.int_to_point(traj_n_12[n][0][0])
# for t in traj_n_12[n]:
#     state = t[0]
#     current_state = state
#     current_position = gw_anomaly.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw_anomaly.int_to_point(last_state)
# plt.show()
# #%%
# # dest = (5,5)
# # mid_point = []
# # for i in range(100):
# #     s = gw.int_to_point(i)
# #     dev = deviation_comp(s, dest)
# #     if dev == deviation:
# #         mid_point.append(s)

# # def select_midpoint(self, deviation, dest):
# #     mid_point = []
# #     for i in range(self.n_states):
# #         s = self.int_to_point(i)
# #         dev = self.deviation_comp(s, dest)
# #         if dev == deviation:
# #             mid_point.append(s)
# #         return random.choice(mid_point)
# # mp = [82,57,58,91,38]
# # mps = 100-np.array(mp)
# # traj_a1_vali3 = []
# # for mp in mps:
# #     ground_rA2 = np.array([gw_a.reward(s,mp) for s in range(gw_a.n_states)])
# #     policy_a2,ground_v2 = value_iteration.find_policy(gw_a.n_states, gw_a.n_actions,
# #                         gw_a.transition_probability, ground_rA2, discount, stochastic=True)
# #     traj = gw_a.generate_trajectories(1,
# #                                             trajectory_length,
# #                                             policy_a2,optimal_policy1,fs,d,100-mp,random_start=False)
# #     traj_a1_vali3.append(traj)
# import Dest_cluster as DC
# import convert_gps_to_mdp as process
# userdata = '/Users/cww3/Desktop/GeolifeTrajectories /Data/'+'068' + '/Trajectory/'
# traj_allcluster, sources_allcluster = DC.dest_cluster(userdata)
# bottom_left = (39.872118, 116.332948)
# bottom_right = (39.872118, 116.479633)
# top_left = (40.000042, 116.332948)
# traj_allcluster_gw = []
# for group in traj_allcluster:
#     trajs_gw, _ = process.data_process(group, grid_size,bottom_left, bottom_right, top_left)#,bottom_left, bottom_right, top_left
#     traj_allcluster_gw.append(trajs_gw)
# #%%
# #extract sources from normal trajectories


# sources_gw_1 = []
# dest_gw_1 = []
# for t in traj_allcluster_gw[0]:
#     sources_gw_1.append(t[0][0])
#     dest_gw_1.append(t[-1][0])

# sources_gw_2 = []
# dest_gw_2 = []
# for t in traj_allcluster_gw[1]:
#     sources_gw_2.append(t[0][0])
#     dest_gw_2.append(t[-1][0])

# sources_gw_3 = []
# dest_gw_3 = []
# for t in traj_allcluster_gw[2]:
#     sources_gw_3.append(t[0][0])
#     dest_gw_3.append(t[-1][0])

# sources_gw_4 = []
# dest_gw_4 = []
# for t in traj_allcluster_gw[3]:
#     sources_gw_4.append(t[0][0])
#     dest_gw_4.append(t[-1][0])
    

# #state repeat times for one trajectory ---> so I can randomly smaple from this list
# # res_trajs = []
# # for g in traj_allcluster_gw:
# #     for t in g:
# #         t_states = np.array(t)[:,0]
# #         res = {list(t_states).count(i) for i in t_states}
# #         res_trajs.append(list(res))

# # repeat_times = [i for t in res_trajs for i in t]
# # import scipy.stats.distributions
# # mean, var  = scipy.stats.distributions.norm.fit(repeat_times)
# irl = [0.807,0.831,0.864,0.853]
# import matplotlib.pyplot as plt
# ocsvm = [0.264,0.260,0.270,0.276]
# isf = [0.430,0.5,0.609,0.663]
# lof = [0.233,0.258,0.256,0.269]
# D = [8,12,16,20]
# plt.plot(D,ocsvm,'-o',label='One Class SVM')
# plt.plot(D,isf,'-o',label='Isolation Forest')
# plt.plot(D,lof,'-o',label='Local Outlier Factor')
# plt.plot(D,irl,'-o',label='IRL(ours)')
# plt.xlabel('Deviation')
# plt.ylabel('F1(mean)')
# plt.legend()
# plt.show()

# # import random
# # random.choice(repeat_times)

# # grid_size = 50
# # discount = 0.95
# # wind = 0
# # gw = Gridworld_anomaly(grid_size, wind, discount)
# # deviation = 2
# repeat_times=[1,2,3,3,4,4,5]
# #%% generate anomalies
# sources_gw = [sources_gw_1, sources_gw_2, sources_gw_3]#, sources_gw_4]
# dest_gw = [dest_gw_1, dest_gw_2, dest_gw_3]#, dest_gw_4]
# traj_dev8 = []
# traj_dev12 = []
# traj_dev16 = []
# traj_dev20 = []
# for i in range(len(traj_allcluster_gw)):
#     anomaly_8 = gw.generate_trajectories(41,sources_gw[i],dest_gw[i],8,repeat_times)
#     traj_dev8.append(anomaly_8)
#     anomaly_12 = gw.generate_trajectories(41,sources_gw[i],dest_gw[i],12,repeat_times)
#     traj_dev12.append(anomaly_12)
#     anomaly_16 = gw.generate_trajectories(41,sources_gw[i],dest_gw[i],16,repeat_times)
#     traj_dev16.append(anomaly_16)
#     anomaly_20 = gw.generate_trajectories(41,sources_gw[i],dest_gw[i],20,repeat_times)
#     traj_dev20.append(anomaly_20)
    

# traj_normal_0 = gw_generate.generate_detour(41,traj_allcluster_gw[0],0,repeat_times,0)
# traj_normal_1 = gw_generate.generate_detour(41,traj_allcluster_gw[1],0,repeat_times,0)
# traj_normal_2 = gw_generate.generate_detour(41,traj_allcluster_gw[2],0,repeat_times,0)
# # traj_dev8 = gw.generate_trajectories(41,sources_gw,dest_gw,8,repeat_times)
# # traj_dev12 = gw.generate_trajectories(41,sources_gw,dest_gw,12,repeat_times)
# # traj_dev16 = gw.generate_trajectories(41,sources_gw,dest_gw,16,repeat_times)
# # traj_dev20 = gw.generate_trajectories(41,sources_gw,dest_gw,20,repeat_times)
# """
# generate detour anomalies
# 1.select a source and a destination
# 2.select a mid point based on deviation
# 3.use mid point to generate policy 1
# 4.use destination to generate policy 2
# 5.generate anomalous trajectory (including choose repeat times)
# """
# #%%
# plt.pcolor(rewards.reshape((grid_size, grid_size)))
# plt.colorbar()
# num=7
# last_position = gw.int_to_point(traj_dev16[0][num][0][0])
# for t in traj_dev16[0][num]:
#     state = t[0]
#     current_state = state
#     current_position = gw.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw.int_to_point(last_state)
# plt.show()



# # with open("file_d8.txt", "w") as output:
# #     output.write(str(traj_dev8))
# # with open("file_d12.txt", "w") as output:
# #     output.write(str(traj_dev12))
# # with open("file_d16.txt", "w") as output:
# #     output.write(str(traj_dev16))
# # with open("file_d20.txt", "w") as output:
# #     output.write(str(traj_dev20))





# #%%
# """
# generate perturbed anomalies
# by adding guassian noise
# """

# sources_gw = []
# for t in trajs_gw:
#     sources_gw.append(t[0][0])

# anomaly_rw = gw.generate_ramdomwalk(10,sources_gw,rewards,0.6)
# plt.pcolor(rewards.reshape((grid_size, grid_size)))
# plt.colorbar()
# last_position = gw.int_to_point(anomaly_rw[4][0][0])
# for t in anomaly_rw[4]:
#     state = t[0]
#     current_state = state
#     current_position = gw.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw.int_to_point(last_state)
# plt.show()


# noise = np.random.normal(0,0.005,(len(group_a[7]),2))

# traj_noise = group_a[7] + noise
# traj_noise_gw, _ = process.data_process([traj_noise], grid_size,bottom_left, bottom_right, top_left)

# plt.pcolor(rewards.reshape((grid_size, grid_size)))
# plt.colorbar()
# last_position = gw.int_to_point(traj_noise_gw[0][0][0])
# for t in traj_noise_gw[0]:
#     state = t[0]
#     current_state = state
#     current_position = gw.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw.int_to_point(last_state)
# plt.show()


# plt.pcolor(rewards.reshape((grid_size, grid_size)))
# plt.colorbar()
# last_position = gw.int_to_point(trajs_gw[7][0][0])
# for t in trajs_gw[7]:
#     state = t[0]
#     current_state = state
#     current_position = gw.int_to_point(current_state)
#     #print(last_position)
#     #print(current_position)
#     plt.annotate("", xy=tuple(np.add(current_position,(0.5,0.5))), xytext=tuple(np.add(last_position,(0.5,0.5))),
#             arrowprops=dict(arrowstyle="->",facecolor='black'))
#     last_state = state
#     last_position = gw.int_to_point(last_state)
# plt.show()

#perturbation: gw to gps (using middle points in the grid), add noise, then to gw again

