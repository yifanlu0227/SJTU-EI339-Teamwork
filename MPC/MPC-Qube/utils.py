import torch
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = torch.tensor(datas)
        self.labels = torch.tensor(labels)

    def __getitem__(self, index):  # return tensor
        datas, target = self.datas[index], self.labels[index]
        return datas, target

    def __len__(self):
        return len(self.datas)

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def print_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        config = yaml.load(f)
        print("************************")
        print("*** model configuration ***")
        print(yaml.dump(config["model_config"], default_flow_style=False, default_style=''))
        print("*** train configuration ***")
        print(yaml.dump(config["training_config"], default_flow_style=False, default_style=''))
        print("************************")
        print("*** dataset configuration ***")
        print(yaml.dump(config["dataset_config"], default_flow_style=False, default_style=''))
        print("************************")
        print("*** MPC controller configuration ***")
        print(yaml.dump(config["mpc_config"], default_flow_style=False, default_style=''))
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def anylize_env(env, test_episodes = 100,max_episode_step = 500, render = False):
    print("state space shape: ", env.observation_space.shape)
    print("state space lower bound: ", env.observation_space.low)
    print("state space upper bound: ", env.observation_space.high)
    print("action space shape: ", env.action_space.shape)
    print("action space lower bound: ", env.action_space.low)
    print("action space upper bound: ", env.action_space.high)
    print("reward range: ", env.reward_range)
    rewards = []
    steps = []
    for episode in range(test_episodes):
        env.reset()
        step = 0
        episode_reward = 0
        for _ in range(max_episode_step):
            if render:
                env.render()
            step += 1
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            print(next_state)
            episode_reward += reward
            if done:
               # print("done with step: %s " % (step))
                break
        steps.append(step)
        rewards.append(episode_reward)
    env.close()
    print("Randomly sample actions for %s episodes, with maximum %s steps per episodes"
          % (test_episodes, max_episode_step))
    print(" average reward per episode: %s, std: %s " % (np.mean(rewards), np.std(rewards) ))
    print(" average steps per episode: ", np.mean(steps))
    print(" average reward per step: ", np.sum(rewards)/np.sum(steps))

def min_max_scaler(d_in):  # scale the data to the range [0,1]
    d_max = np.max(d_in)
    d_min = np.min(d_in)
    d_out = (d_in - d_min) / (d_max - d_min)
    return d_out, d_min, d_max


