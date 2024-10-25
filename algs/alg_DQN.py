from alg_DQN_supplementary import *


def main():

    # PARAMETERS
    episodes = 10
    # render_mode="rgb_array"
    render_mode = "human"

    # PREPARATIONS
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())

    # env = gym.make("LunarLander-v3", render_mode="human")
    # env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
    # env = gym.make('CliffWalking-v0', render_mode="human")
    # env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    env = gym.make("CartPole-v1", render_mode=render_mode)

    for episode in range(episodes):
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

    env.close()


if __name__ == '__main__':
    main()