import random
import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from .arms import Arms

class StochasticArms(Arms, ABC):
    def __init__(self, number_of_arms):
        super().__init__(number_of_arms)
        self.distrib_parameters = self._generate_parameters()
        self.arms_expectation = self._get_arms_expectation()
        
    def get_best_arm_cumul_rewards(self, horizon):
        return np.cumsum(np.full(horizon, np.max(self.arms_expectation)))

    @abstractmethod
    def _generate_parameters(self):
        pass
    
    @abstractmethod
    def _get_arms_expectation(self):
        pass

    @abstractmethod
    def plot_distrib(self):
        pass
    
class BernouilliArms(StochasticArms):
    def _generate_parameters(self):
        return np.random.uniform(size=self.number_of_arms)
        
    def _get_arms_expectation(self):
        return self.distrib_parameters

    def pull_arm(self, arm_id):
        return np.random.binomial(1, self.distrib_parameters[arm_id])
    
    def plot_distrib(self):
        plt.plot(self.distrib_parameters, 'o')
        plt.xticks(range(len(self.distrib_parameters)))
    
class BetaArms(StochasticArms):
    def _generate_parameters(self):
        return np.random.uniform(low=0.001, high=10, size=(self.number_of_arms, 2)) # alpha and beta

    def _get_arms_expectation(self):
        return self.distrib_parameters[:, 0]/(self.distrib_parameters[:, 0] + self.distrib_parameters[:, 1])
        
    def pull_arm(self, arm_id):
        return np.random.beta(self.distrib_parameters[arm_id, 0], self.distrib_parameters[arm_id, 1])
    
    def plot_distrib(self):
        x = np.linspace(0.05, 0.95, 5000)
        for param in self.distrib_parameters:
            y = beta.pdf(x, param[0], param[1])
            plt.plot(x, y)        