import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


iter = []
re = []
total = 0

class Easy21():
    def draw_card(self):
        # ---Return---
        # value : card value
        # operation: 1 -> add, -1 -> subtract
        value = np.random.randint(1, 11)
        tmp = np.random.rand()

        if tmp > 1 / 3:
            return (value, 1)
        else:
            return (value, -1)

    def draw_one_card(self, score):
        value, operation = self.draw_card()
        score += value * operation
        return score

    def dealer_turn(self, score):
        while (1 <= score < 16):  # only case dealer continue to hit
            score = self.draw_one_card(score)
        return score

    def go_bust(self, score):
        return (score > 21 or score < 1)

    # def self.state_after_action(self, state, action):
    #     if action == 0:
    #         state[1] = self.draw_one_card(self, state[1])
    #     else:
    #         continue

    def step(self, state, action):
        # ---Input---
        # state: tuple, (dealer's first card, player's current sum)
        # action: int, 1 -> "stick", 0 -> "hit"
        #
        # ---Return---
        # next_state: tuple,
        # reward: int,
        # terminal: bool,

        dealer_score, player_score = state
        player_action = action

        if action == 1:  # stick
            dealer_score = self.dealer_turn(dealer_score)
            terminal = True
            next_state = (dealer_score, player_score)  # dealer's score may out of range
            if (self.go_bust(dealer_score)):  # dealer go bust
                reward = 1
            elif (dealer_score == player_score):  # two sums are the same.
                reward = 0
            else:
                reward = int((player_score - dealer_score) / np.abs(
                    player_score - dealer_score))  # if player's sum is larger, then reward = 1

        else:  # hit
            player_score = self.draw_one_card(player_score)
            next_state = (dealer_score, player_score)
            if (self.go_bust(player_score)):
                reward = -1
                terminal = True
            else:
                reward = 0
                terminal = False

        return next_state, reward, terminal


class Policy_Iteration():


    def __init__(self,n_iter = 100000):
        self.P = self.__init_P_table()
        self.n_iteration = n_iter
        # self.alpha = 0.15
        # #alpha
        self.gamma = 1
        #discount factor gamma = 1
        # epsilon means in
        self.V = self.__init_V_table()
        self.win_times = 0

    def __init_P_table(self):
        dealer = np.arange(1, 11)
        player = np.arange(1, 22)
        states = [(d, p) for d in dealer for p in player]
        P_table = {}
        for state in states:
            P_table[state] = [1,0]
        # all hit
        return P_table

    def __init_V_table(self):
        # In fact,the value table is depending on the policy
        dealer = np.arange(1, 11)
        player = np.arange(1, 22)
        states = [(d, p) for d in dealer for p in player]
        V_table = {}
        for state in states:
            V_table[state] = 0
            if state[0] > state[1]:
                V_table[state] = -99
            elif state[0] < state[1]:
                V_table[state] = 1
            elif state[0] == state[1] :
                V_table[state] = 0
        return V_table

    def episode(self):
        self.game = Easy21()
        dealer_score = np.random.randint(1, 11)
        player_score = np.random.randint(1, 11)
        state = (dealer_score, player_score)
    #    print(state)
        self.possible_actions = [0,1]
        totalreward = 0
        terminal = False
        global total


        while not terminal:
            #policy evaluation
            #solve for V(s)
            all_states = set()
            for action in self.possible_actions:
                tmp = self.game.step(state,action)
                print(tmp)
                next_state = tmp[0]
                all_states.add(next_state)
                reward = tmp[1]
                totalreward += reward
                terminal = tmp[2]

                next_value = self.V[state]
                # Belleman equation
                print (self.P[state][action])
                self.V[state] += (self.P[state][action] *

                          (reward + self.gamma * next_value))
            #policy improvement
            #Choose the best action and renew the policy
            #next_policy = self.P

            for state in all_states:

                value = -9999

                max_index = []

                result = [0,0]  # initialize the policy

                # for every actions, calculate

                # [reward + (discount factor) * (next state value function)]

                for index, action in enumerate(self.possible_actions):
                    next_state = tmp[0]
                    reward = tmp[1]
                    terminal = tmp[2]

                    if not terminal:
                        temp = reward + self.gamma * self.V[next_state]

                    # We normally can't pick multiple actions in greedy policy.

                    # but here we allow multiple actions with same max values

                        if temp == value:

                            max_index.append(index)

                        elif temp > value:
                            print ("yes")

                            value = temp

                            max_index.clear()

                            max_index.append(index)

                        # probability of action
                try:
                    prob = 1 / len(max_index)

                    for index in max_index:
                        self.P[state][index] = prob
                except:
                    continue
        print("totalreward", totalreward)
        if totalreward == 1:
            print("win!")
            self.win_times += 1
        total += totalreward
        print(total)
        re.append(total)


    def train(self):
        for i in range(self.n_iteration):
            print(f"iter {i}")
            iter.append(i)
            self.episode()



policy_iteration = Policy_Iteration()

policy_iteration.train()

res = []
print("wim times", policy_iteration.win_times)
for i in range(2, len(iter)):
    print(iter[i], re[i])
    print(re[i]/iter[i])
    res.append(re[i]/iter[i])
plt.plot(iter[2:], res, color='red', linewidth=1.0, linestyle='-')
plt.xlabel("episode")
plt.ylabel("average reward")
plt.show()
