import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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


class Agent():
    def __init__(self):
        self.s_len = 10 * 21   # num of states
        self.a_len = 2   # either stick or hit
        self.r = []   # reward for every state
        self.pi = np.array([0 for s in range(self.s_len)])   # action policy for every state
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)  # transition matrix for actions

        for i , action in enumerate()



class policy_iteration():
    def policy_evaluation(self):

