import numpy as np
from Hive import Hive
from Hive import Utilities
import SimAnneal as SA
import RandomShooting as RS

class MPC(object):
    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]
        self.evaluator = Evaluator(self.gamma)

    def act(self, state, dynamic_model):
        '''
        Optimize the action by Artificial Bee Colony algorithm
        :param state: (numpy array) current state
        :param dynamic_model: system dynamic model
        :return: (float) optimal action
        '''
        self.evaluator.update(state, dynamic_model)
        # optimizer = Hive.BeeHive( lower = [[float(self.action_low),float(self.action_low)]] * self.horizon,
        #                           upper = [[float(self.action_high),float(self.action_high)]]* self.horizon,
        #                           fun = self.evaluator.evaluate,
        #                           numb_bees = self.numb_bees,
        #                           max_itrs = self.max_itrs,
        #                           verbose=False)
        # optimizer = SA.SimAnneal(
        #     lower=[[float(self.action_low),float(self.action_low)]] * self.horizon,
        #     upper=[[float(self.action_high),float(self.action_high)]] * self.horizon,
        #     fun=self.evaluator.evaluate,
        #     scale=3e-2)
        optimizer = RS.RandomShooting(
            lower=[[float(self.action_low),float(self.action_low)]] * self.horizon,
            upper=[[float(self.action_high),float(self.action_high)]] * self.horizon,
            fun=self.evaluator.evaluate,
            N=50)
        cost = optimizer.run()
        # print("Solution: ",optimizer.solution[0])  # 0.32071... 有正有负
        # print("Fitness Value ABC: {0}".format(optimizer.best))  #  -0.000903...
        # Uncomment this if you want to see the performance of the optimizer
        # Utilities.ConvergencePlot(cost)
        return optimizer.solution[0]

class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma
        self.Q = np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2])  # see dim of state space
        self.R = np.diag([1e-4, 1e-4])  # see dim of action space
        self.min_rew = 1e-4

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions): # actions: 对于H-steps的H个action。传入这个函数的时候实际上还存着self.state
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate( (state_tmp,actions[j]) )
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0] # 前面使用了unsqueeze(0) 所以这里取state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j]) # gamma衰减系数
        return rewards # -1.1281110966357758e-05

    def get_reward(self,obs,action_n):
        self._state_des = np.zeros(obs.shape)
        err_s = (self._state_des - obs).reshape(-1, )
        err_a = action_n.reshape(-1, )
        quadr_cost = err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))

        obs_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, 4.*np.pi, 4.*np.pi, 0.5, 0.5]).reshape(-1, )
        act_max = np.array([5.0, 5.0]).reshape(-1,)

        max_cost = obs_max.dot(self.Q.dot(obs_max)) + act_max.dot(self.R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        self.c_max = -1.0 * np.log(self.min_rew) / max_cost
        # Calculate the scaled exponential
        rew = np.exp(-self.c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
        return float(rew)

    # 下面是BallBalancerSim的源码reward函数
    # def _rew_fcn(self, obs, action):
    #     err_s = (self._state_des - obs).reshape(-1,)  # or self._state
    #     err_a = action.reshape(-1,)
    #     quadr_cost = err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))
    #
    #     obs_max = self.state_space.high.reshape(-1, )
    #     act_max = self.action_space.high.reshape(-1, )
    #
    #     max_cost = obs_max.dot(self.Q.dot(obs_max)) + act_max.dot(self.R.dot(act_max))
    #     # Compute a scaling factor that sets the current state and action in relation to the worst case
    #     self.c_max = -1.0 * np.log(self.min_rew) / max_cost
    #
    #     # Calculate the scaled exponential
    #     rew = np.exp(-self.c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
    #     return float(rew)

