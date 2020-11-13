import argparse
from itertools import count

import gym
import scipy.optimize
import os
import pickle

import quanser_robots
import torch
from tqdm import tqdm, trange
from models import *
import time
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()

env = gym.make(args.env_name)

is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
print("policy_net:", policy_net)
print("value_net:", value_net)

episode_reward = []
dtype = torch.float64

# for i_episode in trange(args.max_iter_num):

state = env.reset()
state = running_state(state)
reward_sum = 0
episode_reward.append(reward_sum)

for i in range(10):
    state = env.reset()
    state = running_state(state)

    for t in count(1):  # Don't infinite loop while learning
        state_var = tensor(state).unsqueeze(0).to(dtype)
        action = policy_net(state_var)[0][0].detach().numpy()
        action = int(action) if is_disc_action else action.astype(np.float64)
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        next_state = running_state(next_state)

        if args.render:
            env.render()
        if done:
            break

        state = next_state

        # print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_sum))
    time.sleep(5)
    env.close()
