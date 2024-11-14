######
#	gym.openai.com/docs/
######

import gym

# 创建一个 'CartPole-v0' 环境
env = gym.make( 'CartPole-v0' )	

# 循环运行20次实验
for i_episode in range(20):
    # 初始化环境
	state = env.reset()

	# 每个实验循环运行1000个时间步
	for t in range(1000):
		# env.render()
		print(state)
  
		# 从动作空间中随机采样一个动作
		action = env.action_space.sample()
  
		# 执行动作并返回新的状态、奖励、完成标志和其他信息
		state, reward, done, _ = env.step(action)

		if done:
			# 如果任务完成（例如小车失去平衡），结束本次实验
			print('Episode #%d finished after %d timesteps' % (i_episode + 1, t))
			break

