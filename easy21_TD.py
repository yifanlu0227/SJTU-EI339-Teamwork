import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Easy21():
	def draw_card(self):
		# ---Return---
		# value : card value
		# operation: 1 -> add, -1 -> subtract
		value = np.random.randint(1,11)
		tmp = np.random.rand()

		if tmp>1/3:
			return (value,1)
		else:
			return (value,-1)

	def draw_one_card(self,score):
		value, operation = self.draw_card()
		score += value * operation
		return score

	def dealer_turn(self,score):
		while(1<=score<16): # only case dealer continue to hit
			score = self.draw_one_card(score)
		return score

	def go_bust(self,score):
		return (score>21 or score<1)

	def step(self,state,action):
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

		if action == 1: # stick
			dealer_score = self.dealer_turn(dealer_score)
			terminal = True
			next_state = (dealer_score,player_score) # dealer's score may out of range
			if(self.go_bust(dealer_score)): # dealer go bust
				reward = 1
			elif(dealer_score==player_score): # two sums are the same.
				reward = 0
			else:
				reward = int((player_score-dealer_score)/np.abs(player_score-dealer_score)) # if player's sum is larger, then reward = 1

		else: # hit
			player_score = self.draw_one_card(player_score)
			next_state = (dealer_score,player_score)
			if(self.go_bust(player_score)):
				reward = -1
				terminal = True
			else:
				reward = 0
				terminal = False
			
		return next_state,reward,terminal

class Q_Learning():
	def __init__(self,n_iter=100000):
		# self.Q = np.zeros((11,22,2))  #only use 1-10,1-21
		self.Q = self.init_Q_table()
		self.n_iteration = n_iter
		self.alpha = 0.1
		self.gamma = 1
		self.epsilon = 0.2

	def init_Q_table(self):
		dealer = np.arange(1,11)
		player = np.arange(1,22)
		states = [(d,p) for d in dealer for p in player]
		Q_table = {}
		for state in states:
			Q_table[state] = np.zeros(2)
		return Q_table

	def episode(self):
		game = Easy21()
		dealer_score = np.random.randint(1,11)
		player_score = np.random.randint(1,11)
		state = (dealer_score,player_score)
		terminal = False
		while not terminal:	
			# get action by ε-greedy
			tmp = np.random.rand()
			if tmp < self.epsilon:
				action = np.random.randint(2)    # actually a new action at time t+1
			else:
				action = np.argmax(self.Q[state])

			state_new,reward,terminal = game.step(state,action)

			self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + \
											self.alpha*(reward+self.gamma*max(self.Q.get(state_new,(0,))))
			print(f"state: {state} \t action: {action} \t reward: {reward} \t value: {self.Q[state][action]}")
			state = state_new

	def train(self):
		for i in range(self.n_iteration):
			print(f"iter {i}")
			self.episode()

	def plot(self):
		X = np.arange(1, 11)
		Y = np.arange(1, 22)
		Z = np.zeros((21, 10))

		for key in self.Q:
			x = key[0]
			y = key[1]
			Z[y - 1][x - 1] = np.max(self.Q[key])

		X, Y = np.meshgrid(X, Y)
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
		ax.set_xlabel('Dealer showing')
		ax.set_ylabel('Play sum')
		ax.set_zlabel('value')
		plt.title("Q learning")
		plt.savefig("result.jpg")
		plt.show()



q_learning = Q_Learning()
q_learning.train()
q_learning.plot()
