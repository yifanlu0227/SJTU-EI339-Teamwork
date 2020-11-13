import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import xlwt
from tqdm import trange

iter = []
re = []
totalreward=0


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
	def __init__(self, n_iter=100000, alpha=0.05, epsilon=0.1):
		# self.Q = np.zeros((11,22,2))  #only use 1-10,1-21
		self.Q = self.init_Q_table()
		self.n_iteration = n_iter
		self.alpha = alpha # 0.05
		self.gamma = 1
		self.epsilon = epsilon # 0.1
		self.test_num = 10000
		self.test_win = 0

	def init_Q_table(self):
		dealer = np.arange(1, 11)
		player = np.arange(1, 22)
		states = [(d, p) for d in dealer for p in player]
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
			# print(f"state: {state} \t action: {action} \t reward: {reward} \t value: {self.Q[state][action]}")
			global totalreward
			totalreward += reward
			re.append(totalreward)
			state = state_new

	def episode_test(self):
		game = Easy21()
		dealer_score = np.random.randint(1, 11)
		player_score = np.random.randint(1, 11)
		state = (dealer_score, player_score)
		terminal = False
		while not terminal:	
			# get action by ε-greedy
			tmp = np.random.rand()
			if tmp < self.epsilon:
				action = np.random.randint(2)    # actually a new action at time t+1
			else:
				action = np.argmax(self.Q[state])

			state_new, reward, terminal = game.step(state,action)

			state = state_new

		if(reward==1):
			self.test_win += 1

	def train(self):
		for i in range(self.n_iteration):
			# print(f"iter {i}")
			iter.append(i)
			self.episode()

	def test(self):
		for i in range(self.test_num):
			self.episode_test()

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
		plt.savefig("result_TD.jpg")
		plt.show()


q_learning = Q_Learning()
q_learning.train()
res = []

alpha_pool = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
epsilon_pool = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

# for i in range(2, len(iter)):
# 	print(iter[i], re[i])
# 	print(re[i]/iter[i])
# 	res.append(re[i]/iter[i])
# plt.plot(iter[2:], res, color='red', linewidth=1.0, linestyle='-')
# plt.xlabel("episode")
# plt.ylabel("average reward")
# plt.show()
# q_learning.plot()
# q_learning.test()
# print("In {} test, the agent wins {} times.".format(q_learning.test_num, q_learning.test_win))

win_ratio = []
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
style = xlwt.XFStyle() # 初始化样式
font = xlwt.Font() # 为样式创建字体
font.name = 'Times New Roman'
font.bold = True # 黑体
font.underline = True # 下划线
font.italic = True # 斜体字
style.font = font # 设定样式

for i in trange(len(alpha_pool)):
	for j in range(len(epsilon_pool)):
		q_learning = Q_Learning(100000, alpha_pool[i], epsilon_pool[j])
		q_learning.train()
		q_learning.test()
		print(q_learning.test_win/q_learning.test_num)
		print(q_learning.test_win, q_learning.test_num)
		worksheet.write(i, j, q_learning.test_win/q_learning.test_num)
		win_ratio.append(q_learning.test_win/q_learning.test_num)

workbook.save('result_TD.xls')
print(win_ratio)
