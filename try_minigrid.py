import gym
import minigrid
# import gym_minigrid

# env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
# env = gym.make("MiniGrid-SimpleCrossingS11N5-v0", render_mode="human")
env = gym.make("MiniGrid-KeyCorridorS3R1-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   # action = policy(observation)  # User-defined policy function
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

for k in gym.envs.registry.keys():
    print(f'{k}')
# print(gym.envs.registry.keys())
