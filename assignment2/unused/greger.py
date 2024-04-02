import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

epsilon = 0.99
total_episodes = 10000
max_steps = 200
alpha = 0.1
gamma = 0.95

n_actions = env.action_space.n
state = env.reset()[0]  # Extract only the state array
n_observations = len(state)

# Example of discretizing the state space
n_bins = 100 

"""
n_bins må økes!!!! 


og grensene på binsene må også endres, se bilde i discord
"""


bins = [np.linspace(-4.8, 4.8, n_bins),  # Cart position
        np.linspace(-50.0, 50.0, n_bins),  # Cart velocity
        np.linspace(-0.418, 0.418, n_bins),  # Pole angle
        np.linspace(-50.0, 50.0, n_bins)]  # Pole velocity

print("Initial state:", state)
print("Type of state:", type(state))

def to_discrete_state(state, bins):
    discrete_state = []
    for i in range(len(state)):
        # Ensures state[i] is numerical value, not array or dict
        if isinstance(state[i], (int, float, np.number)):
            bin_index = np.digitize(state[i], bins[i])
            discrete_state.append(bin_index - 1)
        else:
            raise ValueError(f"state[{i}] must be a numerical value, got {type(state[i])} with value {state[i]} instead.")
    return discrete_state

# Calculate total number of discrete states
total_states = tuple(len(b) for b in bins)

# Initialize Q-table
Q = np.zeros(total_states + (n_actions,))
#print(Q.shape)

def choose_action(discrete_state):
    if np.random.uniform(0, 1) < epsilon:
        #print("Calculated")
        return np.argmax(Q[tuple(discrete_state)])
    else:
        #print("random")
        return env.action_space.sample()

# SARSA Update
def update_sarsa(state, state2, reward, action, action2):
    predict = Q[state, action]
    # Convert state2 to a tuple if it's not already a tuple
    if not isinstance(state2, tuple):
        state2 = (state2,)
    target = reward + gamma * Q[state2 + (action2,)]
    if not isinstance(state, tuple):
        state = (state,)
    Q[state + (action,)] += alpha * (target - predict)

# Q-learning Update
def update_q_learning(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2])
    Q[state, action] += alpha * (target - predict)

training_reward = []

for episode in range(total_episodes):
    t = 0
    state1, _dictionary = env.reset()
    discrete_state1 = to_discrete_state(state1, bins)
    #print(discrete_state1)
    action1 = choose_action(discrete_state1)
    #print(action1)
    total_reward = 0

    while t < max_steps:
        # Execute action
        # unpacking the returned values
        state2, reward, done, info, _ = env.step(action1)

        discrete_state2 = to_discrete_state(state2, bins)

        # Choose the next action (for SARSA)
        action2 = choose_action(discrete_state2)
        #print(f"action 2 {action2}")

        # Update Q-table (SARSA)
        update_sarsa(discrete_state1, discrete_state2, reward, action1, action2)

        # Update state and action
        discrete_state1 = discrete_state2
        action1 = action2

        # Update total reward
        total_reward += reward

        t += 1
        if done:
            break
    training_reward.append(total_reward)

plt.plot(training_reward)
plt.show()