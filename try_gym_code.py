import warnings
# Ignore all DeprecationWarnings from NumPy
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1", render_mode="human")
nb_actions = env.action_space.n
env.observation_space.sample()

#https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187

nb_chunks = 50
cart_pos = np.linspace(-4.8, 4.8, nb_chunks)
cart_vel = np.linspace(-20, 20, nb_chunks) # Actually -inf and inf
pole_ang = np.linspace(-0.418, 0.418, nb_chunks)
pole_vel = np.linspace(-20, 20, nb_chunks) # Actually -inf and inf

observation_space = [cart_pos, cart_vel, pole_ang, pole_vel]

q_table_shape = cart_pos.shape[0], cart_vel.shape[0], pole_ang.shape[0], pole_vel.shape[0], nb_actions
q_table = np.zeros(q_table_shape)

def continous_to_discrete(state, observation_space):
    discrete = []

    for value, space_chunk in zip(state, observation_space):
        # using np.digitize to find in which chunk of cart_pos or pole_ang the value belongs to
        ind = np.digitize(value, space_chunk)
        discrete += [ind - 1] # append the chunk indices of the state
    
    return tuple(discrete)

exploration_prob = 1
exploration_decay = 0.95
min_exploration_prob = 0.01
discount_factor = 0.9

learning_rate = 0.15

episodes = 100
max_episode_iter = 1000
rewards_per_episode = []
for e in range(episodes):

    state, empty_dict = env.reset()
    discrete_state = continous_to_discrete(state, observation_space)
    done = False

    total_reward = 0

    for _ in range(max_episode_iter):
        if np.random.uniform(0,1) < exploration_prob:
            action = env.action_space.sample()
        else:
            # Qtable must be implemented to determine action
            action = np.argmax(q_table[discrete_state])

        next_state, reward, done, truncated, info = env.step(action)

        total_reward += reward
        # Update Qtable

        # I dont understand the updating of the Q table and values
        disc_next_state = continous_to_discrete(next_state, observation_space)
        old_q_value = q_table[discrete_state + (action, )]

        next_q_max = np.max(q_table[disc_next_state])

        new_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * next_q_max)

        q_table[discrete_state + (action,)] = new_q_value
        # Understand the above

        if done:
            break

        state = next_state

    exploration_proba = max(min_exploration_prob, np.exp(-exploration_decay*e))
    rewards_per_episode.append(total_reward)

plt.plot(rewards_per_episode)
plt.show()
