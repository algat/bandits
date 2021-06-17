from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

class Agent(ABC):
    def __init__(self, pull_function, number_of_arms, horizon):
        self.pull_function = pull_function
        self.number_of_arms = number_of_arms
        self.time = 0
        self.rewards = np.zeros(horizon)
        super().__init__()
        
    def _pull_arm(self, arm_id):
        return self.pull_function(arm_id)
        
    def plot_cum_rewards(self, label=None):
        plt.plot(np.cumsum(self.rewards), label=label)
        plt.title("Cumulative rewards")
        
    def plot_regret(self, best_arm_cumul_rewards, label=None):
        regret = best_arm_cumul_rewards - np.cumsum(self.rewards[:self.time])
        plt.plot(regret, label=label)
        plt.title("Regret")
   
    @abstractmethod
    def _update_after_pull(self, arm_id, reward):
        pass
    
    @abstractmethod
    def play(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def plot_knowledge(self):
        pass