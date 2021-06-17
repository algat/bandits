import pytest
import numpy as np

def get_oblivious_reward_pattern_matrix(nb_arms):
    a = np.array([0] + list(range(nb_arms)))
    reward_pattern_matrix = np.eye(nb_arms)[a]
    return reward_pattern_matrix


def test_exp3_oblivious_arms():

    NB_ARMS = 6

    from bandits.arms.adversarial_arms import AdversarialObliviousArm
    reward_pattern_matrix = get_oblivious_reward_pattern_matrix(NB_ARMS)
    arms = AdversarialObliviousArm(number_of_arms=NB_ARMS, horizon=3000, reward_pattern_matrix=reward_pattern_matrix)

    from bandits.agent.adversarial_agent import Exp3Agent

    agent = Exp3Agent(pull_function=arms.pull_arm, number_of_arms=NB_ARMS, horizon=3000)
    for k in range(3000):
        agent.play()
    assert np.argmax(agent.cumul_gain_estimates) == 0


def test_exp3p_non_oblivious_arms():

    NB_ARMS = 4

    from bandits.arms.adversarial_arms import AdversarialNonObliviousArm
    transition_reward_matrix = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    proba_reward_matrix = np.array([[0, 1/3, 1/3, 1/3], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    arms = AdversarialNonObliviousArm(number_of_arms=NB_ARMS, horizon=3000, 
                                    transition_reward_matrix=transition_reward_matrix,
                                    proba_reward_matrix=proba_reward_matrix)

    from bandits.agent.adversarial_agent import Exp3PAgent

    agent = Exp3PAgent(pull_function=arms.pull_arm, number_of_arms=NB_ARMS, horizon=3000)
    for k in range(3000):
        agent.play()
    assert np.argmax(agent.cumul_gain_estimates) == 0