import gymnasium as gym

gym.pprint_registry()

env = gym.make("LunarLander-v3")
validation_env = gym.make('LunarLander-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
    
print(f"状态空间: {state_size}, 动作空间: {action_size}")