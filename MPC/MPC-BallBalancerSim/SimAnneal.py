import numpy as np

class Annealer_Core():
    def __init__(self,lower, upper, fun, scale):
        self.H = len(lower)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.evaluate = fun
        self.scale = scale # 需要调参
        self.actions = self.lower + (self.upper-self.lower)*np.random.rand(*self.lower.shape) # randomly initialize actions on H steps
        self.old_value = self.evaluate(self.actions)
        self.T = 1
        self.T_min = 1e-6
        self.decay = 0.98 # 都是要调的参数

    def move(self):
        # randomly search for every dimension
        # Here state is actions actually.
        actions = self.actions + (self.upper-self.lower)*(np.random.rand(*self.lower.shape)-0.5)*self.scale
        actions = np.minimum(self.upper,actions)
        actions = np.maximum(self.lower,actions)
        return actions,self.evaluate(actions)

    def anneal(self):
        T = self.T
        T_min = self.T_min
        a = self.decay
        while T > T_min:
            new_actions, new_value = self.move()
            if new_value < self.old_value:
                self.old_value = new_value
                self.actions = new_actions
            else:
                p = np.exp((self.old_value - new_value) / T)
                r = np.random.uniform(0, 1)
                if p > r:
                    self.old_value = new_value
                    self.actions = new_actions
            T = T * a
        return self.actions, self.old_value



class SimAnneal():
    def __init__(self, lower, upper, fun, scale=1e-5):
        self.best_value = None
        self.solution = None
        self.lower = lower
        self.upper = upper
        self.fun = fun
        self.scale = scale
        self.MAX_ITER = 30

    def run(self):
        for i in range(self.MAX_ITER):
            sa = Annealer_Core(self.lower,self.upper,self.fun,self.scale) # simulate annealing
            actions, value = sa.anneal()
            if not self.best_value:
                self.solution = actions
                self.best_value = value
            else:
                if value<self.best_value:
                    self.solution = actions
                    self.best_value = value
        self.solution = list(self.solution)
        return self.best_value