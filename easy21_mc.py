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
		self.alpha = 0.05
		self.gamma = 1
		self.epsilon = 0.4
		self.N = self.init_Q_table()

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
		totalreward = 0
		terminal = False

		# The following is the main step for every-visit MC evaluation.
		# For TD, alpha should be set constant, and there is no need to calculate "totalreward"
		# while not terminal:
		# 	state,reward,terminal = game.step(ori_state,ori_action)
		#
		# 	totalreward += reward
		#
		# 	self.N[ori_state][ori_action] += 1
		# 	alpha = 1/self.N[ori_state][ori_action]
		# 	epsilon = 100/(100+np.sum(self.N[ori_state][ori_action]))
		#
		# 	self.Q[ori_state][ori_action] = (1-alpha)*self.Q[ori_state][ori_action] + \
		# 									alpha*(totalreward+self.gamma*max(self.Q.get(state,(0,))))
		# 	print(f"state: {ori_state} \t action: {ori_action} \t reward: {reward} \t value: {self.Q[ori_state][ori_action]}")
		# 	ori_state = state
		#
		# 	tmp = np.random.rand()
		# 	if tmp < epsilon:
		# 		ori_action = np.random.randint(2)    # actually a new action at time t+1
		# 	else:
		# 		ori_action = np.argmax(self.Q)

		trace = []
		while not terminal:

			tmp = np.random.rand()
			if tmp < self.epsilon:
				action = np.random.randint(2)
			else:
				action = np.argmax(self.Q)
			epsilon = 100 / (100 + np.sum(self.N[state][action]))
			
			self.N[state][action] += 1

			trace.append((state, action))

			state, reward, terminal = game.step(state, action)
			totalreward += reward

			# ori_state = state
			# print(f"state: {ori_state} \t action: {ori_action} \t reward: {totalreward} \t value: {self.Q[ori_state][ori_action]}")
		for o_s, o_a in trace:
			alpha = 1/self.N[o_s][o_a]
			# totalreward = totalreward*self.gamma+r
			self.Q[o_s][o_a] += alpha*(totalreward-self.Q[o_s][o_a])

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
		plt.savefig("result_MC_e0.4_0.05.jpg")
		plt.show()



q_learning = Q_Learning()
q_learning.train()
q_learning.plot()
