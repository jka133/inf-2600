from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import ipdb


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

class CartPole2DEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077) 
    that is extended to 2D grid by the definitions described in Gomez and Miikkulainen in 
    ["2-D Pole Balancing with Recurrent Evolutionary Networks"](https://www.cs.utexas.edu/users/nn/downloads/papers/gomez.icann98.pdf)
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1, 2, 3}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    | 2   | Push cart towards top   |
    | 3   | Push cart towards bottom |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space
    The observation is a `ndarray` with shape `(8,)` with the values corresponding to the following positions and velocities:

    | Num | Observation             | Min                 | Max               |
    |-----|-----------------------  |---------------------|-------------------|
    | 0   | Cart Position X         | -4.8                | 4.8               |
    | 1   | Cart Position Y         | -4.8                | 4.8               |
    | 2   | Cart Velocity X         | -Inf                | Inf               |
    | 3   | Cart Velocity Y         | -Inf                | Inf               |
    | 4   | Pole Angle X            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 5   | Pole Angle Y            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 6   | Pole Angular Velocity X | -Inf                | Inf               |
    | 7   | Pole Angular Velocity Y | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments
    ```
    gym.make('CartPole-2D')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4      # Use for y_threshold as well

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, y, y_dot, theta_x, theta_x_dot,  theta_y, theta_y_dot, = self.state

        if action == 1 or action == 0: 
            force = self.force_mag  if action == 1 else -self.force_mag
            costhetax = math.cos(theta_x)
            sinthetax = math.sin(theta_x)
            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_x_dot**2 * sinthetax
            ) / self.total_mass
            thetaacc_x = (self.gravity * sinthetax - costhetax * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costhetax**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc_x * costhetax / self.total_mass
            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta_x = theta_x + self.tau * theta_x_dot
                theta_x_dot = theta_x_dot + self.tau * thetaacc_x
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_x_dot = theta_x_dot + self.tau * thetaacc_x
                theta_x = theta_x + self.tau * theta_x_dot

        else:
            force = self.force_mag  if action == 2 else -self.force_mag
            costhetay = math.cos(theta_y)
            sinthetay = math.sin(theta_y)
            temp_y = (
                force + self.polemass_length * theta_y_dot**2 * sinthetay
            ) / self.total_mass
            thetaacc_y = (self.gravity * sinthetay - costhetay * temp_y) / (
                self.length * (4.0 / 3.0 - self.masspole * costhetay**2 / self.total_mass)
            )
            yacc = temp_y - self.polemass_length * thetaacc_y * costhetay / self.total_mass

            if self.kinematics_integrator == "euler":
                y = y + self.tau * y_dot
                y_dot = y_dot + self.tau * yacc
                theta_y = theta_y + self.tau * theta_y_dot
                theta_y_dot = theta_y_dot + self.tau * thetaacc_y
            else:  # semi-implicit euler
                y_dot = y_dot + self.tau * yacc
                y = y + self.tau * y_dot
                theta_y_dot = theta_y_dot + self.tau * thetaacc_y
                theta_y = theta_y + self.tau * theta_y_dot

        self.state = (x, x_dot, y, y_dot, theta_x, theta_x_dot,  theta_y, theta_y_dot,)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta_x < -self.theta_threshold_radians
            or theta_x > self.theta_threshold_radians
            or y < -self.x_threshold
            or y > self.x_threshold
            or theta_y < -self.theta_threshold_radians
            or theta_y > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(8,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

env = CartPole2DEnv()

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Replay Memory
# 
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
# 
# For this, we're going to need two classes:
# 
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
# 
# 
# 


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Now, let's define our model. But first, let's quickly recap what a DQN is.
# 
# ## DQN algorithm
# 
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
# 
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# $R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t$, where
# $R_{t_0}$ is also known as the *return*. The discount,
# $\gamma$, should be a constant between $0$ and $1$
# that ensures the sum converges. A lower $\gamma$ makes 
# rewards from the uncertain far future less important for our agent 
# than the ones in the near future that it can be fairly confident 
# about. It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
# 
# The main idea behind Q-learning is that if we had a function
# $Q^*: State \times Action \rightarrow \mathbb{R}$, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
# 
# \begin{align}\pi^*(s) = \arg\!\max_a \ Q^*(s, a)\end{align}
# 
# However, we don't know everything about the world, so we don't have
# access to $Q^*$. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# $Q^*$.
# 
# For our training update rule, we'll use a fact that every $Q$
# function for some policy obeys the Bellman equation:
# 
# \begin{align}Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))\end{align}
# 
# The difference between the two sides of the equality is known as the
# temporal difference error, $\delta$:
# 
# \begin{align}\delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))\end{align}
# 
# To minimize this error, we will use the [Huber
# loss](https://en.wikipedia.org/wiki/Huber_loss)_. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of $Q$ are very noisy. We calculate
# this over a batch of transitions, $B$, sampled from the replay
# memory:
# 
# \begin{align}\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)\end{align}
# 
# \begin{align}\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}\end{align}
# 
# ### Q-network
# 
# Our model will be a feed forward  neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing $Q(s, \mathrm{left})$ and
# $Q(s, \mathrm{right})$ (where $s$ is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# ## Training
# 
# ### Hyperparameters and utilities
# This cell instantiates our model and its optimizer, and defines some
# utilities:
# 
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the duration of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #print(sample, eps_threshold)
    if sample > eps_threshold: # I think this is where EXPLOITATION vs EXPLORATION is changed
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1) #Action with highest expected reward (Exploitation)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # Random sample (Exploration)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# ### Training loop
# 
# Finally, the code for training our model.
# 
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes $Q(s_t, a_t)$ and
# $V(s_{t+1}) = \max_a Q(s_{t+1}, a)$, and combines them into our
# loss. By definition we set $V(s) = 0$ if $s$ is a terminal
# state. We also use a target network to compute $V(s_{t+1})$ for
# added stability. The target network is updated at every step with a 
# [soft update](https://arxiv.org/pdf/1509.02971.pdf)_ controlled by 
# the hyperparameter ``TAU``, which was previously defined.


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Below, you can find the main training loop. At the beginning we reset
# the environment and obtain the initial ``state`` Tensor. Then, we sample
# an action, execute it, observe the next state and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
# 
# Below, `num_episodes` is set to 600 if a GPU is available, otherwise 50 
# episodes are scheduled so training does not take too long. However, 50 
# episodes is insufficient for to observe good performance on CartPole.
# You should see the model constantly achieve 500 steps within 600 training 
# episodes. Training RL agents can be a noisy process, so restarting training
# can produce better results if convergence is not observed.

if torch.cuda.is_available():
    num_episodes = 6000
else:
    num_episodes = 200

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        #env.render()
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        print(f"  {int(100*i_episode/num_episodes)}", end="\r")
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break

print('Complete')
#plot_durations(show_result=True)
#plt.ioff()
plt.plot(episode_durations)
plt.show()

# Here is the diagram that illustrates the overall resulting data flow.
# 
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
# 
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. The "older" target_net is also used in optimization to compute the
# expected Q values. A soft update of its weights are performed at every step.

