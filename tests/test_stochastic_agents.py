import pytest
import numpy as np

def custom_pull_arm(arm_id: int) -> float:
    s = np.random.normal(loc=arm_id, scale=0.1, size=1)[0]
    return s

def test_epsilon_greedy_custom_pull_arm():

    NB_ARMS = 10

    from bandits.agent.stochastic_agent import EpsilonGreedyAgent
    agent = EpsilonGreedyAgent(pull_function=custom_pull_arm, number_of_arms=NB_ARMS, horizon=2000)
    for k in range(2000):
        agent.play()
    assert np.argmax(agent.arms_emp_means) == 9

def test_UCB_custom_pull_arm():

    NB_ARMS = 10

    from bandits.agent.stochastic_agent import UCBAgent
    agent = UCBAgent(pull_function=custom_pull_arm, number_of_arms=NB_ARMS, horizon=2000, alpha=100)
    for k in range(2000):
        agent.play()
    print(agent.arms_emp_means)
    assert np.argmax(agent.arms_emp_means) == 9

def test_epsilon_greedy_beta_arms():

    NB_ARMS = 6
    from bandits.arms.stochastic_arms import BetaArms
    from bandits.agent.stochastic_agent import EpsilonGreedyAgent

    arms = BetaArms(number_of_arms=NB_ARMS)

    agent = EpsilonGreedyAgent(pull_function=arms.pull_arm, number_of_arms=NB_ARMS, horizon=2000)
    for k in range(2000):
        agent.play()
    assert np.argmax(arms.arms_expectation) == np.argmax(agent.arms_emp_means)

def test_epsilon_greedy_bernouilli_arms():

    NB_ARMS = 6
    from bandits.arms.stochastic_arms import BernouilliArms
    from bandits.agent.stochastic_agent import UCBAgent

    arms = BernouilliArms(number_of_arms=NB_ARMS)

    agent = UCBAgent(pull_function=arms.pull_arm, number_of_arms=NB_ARMS, horizon=2000, alpha=5)
    for k in range(2000):
        agent.play()
    assert np.argmax(arms.arms_expectation) == np.argmax(agent.arms_emp_means)