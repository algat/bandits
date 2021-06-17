from abc import ABC, abstractmethod

class Arms(ABC):
    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms
        super().__init__()

    @abstractmethod
    def get_best_arm_cumul_rewards(self, horizon):
        pass

    @abstractmethod
    def pull_arm(self, arm_id):
        pass