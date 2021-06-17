from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from .agent import Agent

class AdversarialAgent(Agent, ABC):
    def __init__(self, pull_function, number_of_arms, horizon):
        super().__init__(pull_function, number_of_arms, horizon)
        self.cumul_gain_estimates = np.zeros(number_of_arms)
        probas = np.zeros((number_of_arms, horizon + 1))
        probas[:, 0] = 1 / number_of_arms
        self.probas = probas
        
    def plot_knowledge(self):
        for i in range(self.number_of_arms):
            plt.plot(self.probas[i, :(self.time -1)], label='Arm {}'.format(i))
        plt.legend()
        plt.title("Arms probabilities")
    
    def play(self):
        arm_id = np.random.choice(np.arange(self.number_of_arms), p=self.probas[:, self.time])
        reward = self._pull_arm(arm_id)
        self._update_after_pull(arm_id, reward)
    
    def __str__(self):
        output = " Time: {}, cumulated reward: {}\n".format(self.time, np.sum(self.rewards))
        idx = 0
        for proba in self.probas[:, self.time]:
            output += " arm: {} proba: {}\n".format(idx, proba)
            idx += 1
        return output
        

class Exp3Agent(AdversarialAgent):
    def __init__(self, pull_function, number_of_arms, horizon):
        super().__init__(pull_function, number_of_arms, horizon)
        self.eta = np.sqrt(2*np.log(number_of_arms)/(horizon*number_of_arms))

    def _update_after_pull(self, arm_id, reward):
        self.rewards[self.time] = reward
        self.cumul_gain_estimates[arm_id] += reward/self.probas[arm_id, self.time]

        # to avoid numerical instability
        normalized_cumul_gain_estimates = self.cumul_gain_estimates - np.amin(self.cumul_gain_estimates)
        self.probas[:, self.time + 1] = np.exp(self.eta*normalized_cumul_gain_estimates) / np.sum(np.exp(self.eta*normalized_cumul_gain_estimates))
        self.time += 1


class Exp3PAgent(AdversarialAgent):
    def __init__(self, pull_function, number_of_arms, horizon):
        super().__init__(pull_function, number_of_arms, horizon)
        self.eta = 0.95*np.sqrt(np.log(number_of_arms)/(horizon*number_of_arms))
        self.beta = np.sqrt(np.log(number_of_arms)/(horizon*number_of_arms))
        self.gamma = 1.05*np.sqrt(number_of_arms*np.log(number_of_arms)/horizon)

    def _update_after_pull(self, arm_id, reward):
        self.rewards[self.time] = reward
    
        indicator = np.zeros_like(self.cumul_gain_estimates)
        indicator[arm_id] = 1
        self.cumul_gain_estimates += indicator * reward/self.probas[:, self.time] + self.beta/self.probas[:, self.time]

        # to avoid numerical instability
        normalized_cumul_gain_estimates = self.cumul_gain_estimates - np.amin(self.cumul_gain_estimates)
        self.probas[:, self.time + 1] = (1-self.gamma)*np.exp(self.eta*normalized_cumul_gain_estimates)\
                                         / np.sum(np.exp(self.eta*normalized_cumul_gain_estimates)) + self.gamma/self.number_of_arms
        self.time += 1