from abc import ABC, abstractmethod
import numpy as np
import random
from matplotlib import pyplot as plt
from .agent import Agent


class StochasticAgent(Agent, ABC):
    def __init__(self, pull_function, number_of_arms, horizon):
        super().__init__(pull_function, number_of_arms, horizon)
        self.arms_emp_means = np.zeros(number_of_arms)
        self.arms_counts = np.ones(number_of_arms)

    def _update_after_pull(self, arm_id, reward):
        self.arms_counts[arm_id] += 1
        self.arms_emp_means[arm_id] = self.arms_emp_means[arm_id] + \
                                        (reward - self.arms_emp_means[arm_id])/self.arms_counts[arm_id]
        self.rewards[self.time] = reward
        self.time += 1
        

class EpsilonGreedyAgent(StochasticAgent):
    def __init__(self, pull_function, number_of_arms, horizon, epsilon=0.1):
        super().__init__(pull_function, number_of_arms, horizon)
        self.epsilon = epsilon

    def play(self):
        p = random.uniform(0, 1)
        if p <= self.epsilon:
            arm_id = random.choice(range(self.number_of_arms))
        else:
            arm_id = np.argmax(self.arms_emp_means)
        
        reward = self._pull_arm(arm_id)
        self._update_after_pull(arm_id, reward)

    def __str__(self):
        output = " Time: {}, cumulated reward: {}\n".format(self.time, np.sum(self.rewards))
        idx = 0
        for mean, count in zip(self.arms_emp_means, self.arms_counts):
            output += " arm: {} empirical mean: {} count: {}\n".format(idx, mean, count)
            idx += 1
        return output
    
    def plot_knowledge(self):
        plt.plot(range(len(self.arms_emp_means)), self.arms_emp_means, 'o', color='blue')
        plt.title("Emp mean")
        

class UCBAgent(StochasticAgent):
    def __init__(self, pull_function, number_of_arms, horizon, alpha):
        super().__init__(pull_function, number_of_arms, horizon)
        self.alpha = alpha

    def play(self):
        upper_bounds = self.__get_upper_bounds()
        arm_id = np.argmax(upper_bounds)
        reward = self._pull_arm(arm_id)
        self._update_after_pull(arm_id, reward)
        
    def __str__(self):
        output = " Time: {}, cumulated reward: {}\n".format(self.time, np.sum(self.rewards))
        idx = 0
        for mean, upper_bound, count in zip(self.arms_emp_means, self.__get_upper_bounds(), self.arms_counts):
            output += " arm: {} empirical mean: {} UB: {} count: {}\n".format(idx, mean, upper_bound, count)
            idx += 1
        return output
    
    def plot_knowledge(self):
        upper_bounds = self.__get_upper_bounds()
        plt.plot(range(len(self.arms_emp_means)), self.arms_emp_means, 'o', color='blue')
        for x, lower,upper in zip(range(len(self.arms_emp_means)), 2*self.arms_emp_means - upper_bounds, upper_bounds):
            plt.plot((x,x),(lower,upper), 'ro-', color='orange')
        plt.title("Emp mean and bounds")

    def __get_upper_bounds(self):
        return self.arms_emp_means + np.sqrt(self.alpha * np.log(self.time + 1) / (2 * self.arms_counts))
