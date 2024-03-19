#https://www.geeksforgeeks.org/sarsa-reinforcement-learning/
#https://builtin.com/machine-learning/sarsa
#https://towardsdatascience.com/intro-to-reinforcement-learning-temporal-difference-learning-sarsa-vs-q-learning-8b4184bb4978
#https://github.com/viethoangtranduong/reinforcement-learning/blob/main/SARSA%20vs%20QL/train_SARSA.py

import warnings
# Ignore all DeprecationWarnings from NumPy
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import gym
import matplotlib.pyplot as plt
import math
env = gym.make("CartPole-v1")
N_ACTIONS = env.action_space.n
env.observation_space.sample()

#https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187

N_CHUNKS = 100
cart_pos = np.linspace(-4.8, 4.8, N_CHUNKS)
cart_vel = np.linspace(-20, 20, N_CHUNKS) # Actually -inf and inf
pole_ang = np.linspace(-0.418, 0.418, N_CHUNKS)
pole_vel = np.linspace(-20, 20, N_CHUNKS) # Actually -inf and inf

observation_space = [cart_pos, cart_vel, pole_ang, pole_vel]

Q_TABLE_SHAPE = cart_pos.shape[0], cart_vel.shape[0], pole_ang.shape[0], pole_vel.shape[0], N_ACTIONS
q_table = np.zeros(Q_TABLE_SHAPE)

def continous_to_discrete(state, observation_space):
    #https://ai.stackexchange.com/questions/12255/can-q-learning-be-used-for-continuous-state-or-action-spaces
    discrete = []
    for value, space_chunk in zip(state, observation_space):
        # using np.digitize to find in which chunk of cart_pos or pole_ang the value belongs to
        ind = np.digitize(value, space_chunk)
        discrete += [ind - 1] # append the chunk indi of the state
    return tuple(discrete)

EXPLORATION_PROB = 1 # These variables must be optimized 
ECPLORATION_DECAY = 0.001
MIN_EXPLORATION_PROB = 0.01
#https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning
DISCOUNT_FACTOR = 0.99

LEARNING_RATE = 0.125 # Try different values for this

N_EPISODES = 15000
MAX_EPISODE_LEN = 10000
rewards_per_episode = []
for e in range(N_EPISODES):

    state, unused_dict = env.reset()
    discrete_state = continous_to_discrete(state, observation_space)
    done = False
    total_reward = 0

    for _ in range(MAX_EPISODE_LEN):

        if np.random.random() < EXPLORATION_PROB:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[discrete_state])

        next_state, reward, done, truncated, info = env.step(action)
        discrete_next_state = continous_to_discrete(next_state, observation_space)

        if np.random.random() < EXPLORATION_PROB:
            next_action = env.action_space.sample()
        else:
            # Qtable must be implemented to determine action
            next_action = np.argmax(q_table[discrete_next_state])

        old_q_value = q_table[discrete_state + (action, )]
        next_q_value = q_table[discrete_next_state + (next_action, )]

        # Update Qtable
        new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q_value) # This is the only difference from ql
        q_table[discrete_state + (action, )] = new_q_value

        total_reward += reward
        discrete_state = discrete_next_state
        state = next_state

        if done:
            break

    rewards_per_episode.append(total_reward)
    EXPLORATION_PROB = max(MIN_EXPLORATION_PROB, np.exp(-ECPLORATION_DECAY*e)) # Number of episodes changes the decay
    #exploration_prob = max(min_exploration_prob, exploration_decay*exploration_prob)
    print(f" Episode {e} -- {int(100*e/N_EPISODES)}% finished", end="\r")

# Iterate over each step in rewards_per_episode
meanhundred_rewards_per_episode = []
for i in range(len(rewards_per_episode)):
    if i < 100:
        meanhundred_rewards_per_episode.append(0)  # Append 0 for the first 100 steps
    else:
        # Calculate the mean of the previous 100 rewards
        mean_reward = sum(rewards_per_episode[i-100:i]) / 100
        meanhundred_rewards_per_episode.append(mean_reward)

# Calculate the mean reward for every thousand steps
for i in range(0, len(rewards_per_episode), 1000):
    print(f"Thousand mean step {int(i/1000+1)}: {sum(rewards_per_episode[i:i+1000]) / 1000}  ")

plt.plot(rewards_per_episode)
plt.plot(meanhundred_rewards_per_episode, label="Mean reward per hundred episodes")
plt.xlabel("Episode")
plt.ylabel("Duration (reward)")
plt.legend()
plt.show()

#https://stackoverflow.com/questions/24809757/how-to-make-a-histogram-from-a-list-of-data-and-plot-it-with-matplotlib
counts, bins = np.histogram(rewards_per_episode, bins=500)
plt.stairs(counts, bins, fill=True)
plt.xlabel("Duration (reward)")
plt.ylabel("Frequency of duration")
plt.show()
