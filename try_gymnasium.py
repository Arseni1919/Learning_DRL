import gymnasium as gym

episodes = 10
# render_mode="rgb_array"
render_mode="human"

# env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
# env = gym.make('CliffWalking-v0', render_mode="human")
# env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
env = gym.make("CartPole-v1", render_mode=render_mode)

for ep in range(episodes):
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

env.close()