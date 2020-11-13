# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating
import time
import argparse

# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]

parser = argparse.ArgumentParser(description="demo")
parser.add_argument('-p','--path',type=str,default="config.yml")
args = parser.parse_args()
config_path = args.path

env_id ="BallBalancerSim-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
config = load_config(config_path)
print_config(config_path)

model = DynamicModel(config)

data_fac = DatasetFactory(env,config)
data_fac.collect_random_dataset()

loss = model.train(data_fac.random_trainset,data_fac.random_testset)

mpc = MPC(env,config)

rewards_list = []
for itr in range(config["dataset_config"]["n_mpc_itrs"]):
    t = time.time()
    print("**********************************************")
    print("The reinforce process [%s], collecting data ..." % itr)
    rewards = data_fac.collect_mpc_dataset(mpc, model)
    trainset, testset = data_fac.make_dataset()
    rewards_list += rewards

    plt.close("all")
    plt.figure(figsize=(12, 5))
    plt.title('Reward Trend with %s iteration' % itr)
    plt.plot(rewards_list)
    plt.savefig("storage/reward-" + str(model.exp_number) + ".png")
    print("Consume %s s in this iteration" % (time.time() - t))
    loss = model.train(trainset, testset)
