"""
Implements deep maximum entropy inverse reinforcement learning 
"""

from itertools import product

import numpy as np
import numpy.random as rn
import tf_utils
import value_iteration
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepIRLFC:


  def __init__(self, n_input, lr, n_h1=1024, n_h2=640, n_h3=128, l2=10, name='deep_irl_fc5'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.n_h3 = n_h3
    self.name = name

    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [None, 1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.n_input])
    with tf.variable_scope(name):
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=None)
      fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=None)
      fc3 = tf_utils.fc(fc2, self.n_h3, scope="fc3", activation_fn=tf.nn.relu,
        initializer=None)
      reward = tf_utils.fc(fc3, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, reward, theta


  def get_theta(self):
    return self.sess.run(self.theta)


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms




def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """

  # tf.set_random_seed(1)
  
  N_STATES, N_ACTIONS,_ = np.shape(P_a)

  # init nn model
  nn_r = DeepIRLFC(feat_map.shape[1], lr)

  # find state visitation frequencies using demonstrations
  mu_D = find_feature_expectations(feat_map,trajs)

  # training 
  for iteration in range(n_iters):
    #if iteration % (n_iters/2) == 0:
    print ('iteration: {}'.format(iteration))
    
    # compute the reward matrix
    rewards = nn_r.get_rewards(feat_map)
    rewards = rewards.reshape(-1)
#    
#    # compute policy 
#    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
#    
#    # compute expected svf
#    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    expected_svf = find_expected_svf(N_STATES, rewards, N_ACTIONS, gamma,
                                         P_a, trajs)
    mu_exp = feat_map.T.dot(expected_svf)
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp

    # apply gradients to the neural network
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
    

  rewards = nn_r.get_rewards(feat_map)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)



def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)








def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = rn.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        # print("i: {}".format(i))
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += learning_rate * grad

    return feature_matrix.dot(alpha).reshape((n_states,))

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _ in trajectory:
            svf[int(state)] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _ in trajectory:
            feature_expectations += feature_matrix[int(state)]

    feature_expectations /= len(trajectories)

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = len(trajectories)
    trajectory_length = 30

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy,_ = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount, stochastic=True)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[int(trajectory[0][0])] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy,_ = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd
