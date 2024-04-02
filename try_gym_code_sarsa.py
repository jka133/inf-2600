import numpy as np
import time
import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
N_ACTIONS = env.action_space.n
env.observation_space.sample()

# The structure of the program is inspired by:
#https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
# But it is substansially different since the actions space is continous in this assignment

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
        # https://www.geeksforgeeks.org/python-numpy-np-digitize-method/
        ind = np.digitize(value, space_chunk)
        discrete += [ind - 1] # append the chunk index of the state

    return tuple(discrete)

EXPLORATION_PROB = 1
EXPLORATION_DECAY = 0.001
MIN_EXPLORATION_PROB = 0.01
#https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.125

N_EPISODES = 15000
MAX_EPISODE_LEN = 2000
rewards_per_episode = []

start = time.time()

for e in range(N_EPISODES):

    state, unused_dict = env.reset() # The version of gym used provides an empty dictionary along with the state
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

        # Next action exploration is from: 
        #https://towardsdatascience.com/intro-to-reinforcement-learning-temporal-difference-learning-sarsa-vs-q-learning-8b4184bb4978
        if np.random.random() < EXPLORATION_PROB:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[discrete_next_state])

        old_q_value = q_table[discrete_state + (action, )]
        # The next_q_value is different from Qlearning
        # https://www.geeksforgeeks.org/sarsa-reinforcement-learning/
        # Instead of taking the action with ighest q-value the next action is also on policy
        next_q_value = q_table[discrete_next_state + (next_action, )]

        # Update Qtable
        # The values put into the update is the only difference from qlearning
        new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q_value) 
        q_table[discrete_state + (action, )] = new_q_value

        total_reward += reward

        # "Moving" to the next state
        discrete_state = discrete_next_state
        state = next_state

        if done:
            break

    rewards_per_episode.append(total_reward)
    EXPLORATION_PROB = max(MIN_EXPLORATION_PROB, np.exp(-EXPLORATION_DECAY*e)) # Number of episodes changes the decay
    print(f"\tEpisode {e+1} -- {int(100*e/N_EPISODES)+1}% finished. Time taken {int(time.time() - start)} seconds", end="\r")
print()

# Iterate over each step in rewards_per_episode
meanhundred_rewards_per_episode = [0 for x in range(100)] + [sum(rewards_per_episode[i:i+100])/100 for i in range(N_EPISODES - 100)]

# Calculate the mean reward for every thousand steps
for i in range(0, len(rewards_per_episode), 1000):
    print(f"\tThousand mean step {int(i/1000+1)}: {sum(rewards_per_episode[i:i+1000]) / 1000}  ")

plt.plot(rewards_per_episode)
plt.plot(meanhundred_rewards_per_episode, label="Mean reward per hundred episodes")
plt.xlabel("Episode")
plt.ylabel("Duration (reward)")
plt.title("SARSA")
plt.legend()
plt.show(block=1)

#https://stackoverflow.com/questions/24809757/how-to-make-a-histogram-from-a-list-of-data-and-plot-it-with-matplotlib
trim_size = 0.05
trim_ind = int(N_EPISODES*trim_size)
trimmed_list = sorted(rewards_per_episode)[trim_ind:-trim_ind]
counts, bins = np.histogram(trimmed_list, bins=int(N_EPISODES/100))
plt.stairs(counts, bins, fill=True)
plt.xlabel("Duration (reward)")
plt.ylabel("Frequency of duration")
plt.title("Frequency of duration (trimmed result [0.05-0.95]) SARSA")
plt.show(block=1)
