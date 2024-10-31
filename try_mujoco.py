import gymnasium as gym
import time

# Create the Ant-v4 environment
# env = gym.make('Ant-v5', render_mode="human")
# env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, render_mode="human")
# env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode="human")
env = gym.make('InvertedDoublePendulum-v5', reset_noise_scale=0.1, render_mode="human")

# Reset the environment to get the initial state
obs, info = env.reset()

for _ in range(1000):  # Run for 1000 time steps
    # Take a random action
    action = env.action_space.sample()

    # Step the environment forward and get results
    obs, reward, done, truncated, info = env.step(action)

    # time.sleep(1)

    # Check if the episode is done
    if done or truncated:
        obs, info = env.reset()

# Close the environment properly
env.close()