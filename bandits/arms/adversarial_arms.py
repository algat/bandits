from collections import deque
import numpy as np
from .arms import Arms
from matplotlib import pyplot as plt


class AdversarialObliviousArm(Arms):
    def __init__(self, number_of_arms, horizon, reward_pattern_matrix):
        super().__init__(number_of_arms)
        self.time = 0
        self.reward_pattern_matrix = reward_pattern_matrix
        self.rewards = np.zeros((horizon, number_of_arms))
        self.pattern_length = reward_pattern_matrix.shape[0]

    def get_best_arm_cumul_rewards(self, horizon):
        cum_rewards = np.cumsum(self.rewards, axis=0)
        best_arm_id = np.argmax(cum_rewards[-1,:])
        return cum_rewards[:, best_arm_id]

    def plot_info(self):
        plt.matshow(self.reward_pattern_matrix)

    def _update_state(self):
        state_in_pattern = self.time % self.pattern_length
        self.rewards[self.time, :] = self.reward_pattern_matrix[state_in_pattern, :]
        self.time += 1

    def pull_arm(self, arm_id):
        state_in_pattern = self.time % self.pattern_length
        reward = self.reward_pattern_matrix[state_in_pattern, arm_id]
        self._update_state()
        return reward


class AdversarialNonObliviousArm(Arms):
    def __init__(self, number_of_arms, horizon, transition_reward_matrix, proba_reward_matrix):
        super().__init__(number_of_arms)
        self.time = 0
        self.transition_reward_matrix = transition_reward_matrix
        self.proba_reward_matrix = proba_reward_matrix
        self.rewards = np.zeros((horizon, number_of_arms))
        self.current_reward = np.ones(number_of_arms)
        self.current_proba = np.full(number_of_arms, 1 / number_of_arms)

    def get_best_arm_cumul_rewards(self, horizon):
        cum_rewards = np.cumsum(self.rewards, axis=0)
        best_arm_id = np.argmax(cum_rewards[-1,:])
        return cum_rewards[:, best_arm_id]

    def plot_info(self):
        plt.matshow(self.transition_reward_matrix)
        plt.matshow(self.proba_reward_matrix)
    
    def _update_state(self, arm_id):
        self.rewards[self.time, :] = self.current_reward * self.current_proba
        self.current_reward, self.current_proba = self.transition_reward_matrix[arm_id], self.proba_reward_matrix[arm_id]
        self.time += 1

    def pull_arm(self, arm_id):
        reward, proba = self.current_reward[arm_id], self.current_proba[arm_id]
        self._update_state(arm_id)
        return np.random.binomial(1, p=proba) * reward