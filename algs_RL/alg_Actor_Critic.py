import matplotlib.pyplot as plt

from algs_RL.alg_Actor_Critic_sup import *
from plot_functions import *
from global_functions import *


def main():
    # parameters
    GAMMA = 0.99
    LR = 3e-3
    EPS = np.finfo(np.float32).eps.item()
    running_reward = 10
    reward_threshold = 450
    # render_mode = "human"  # "human" / "rgb_array" / None
    render_mode = None
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # for plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.001
    episode_durations: List[int] = []
    running_rewards: List[float] = []

    # seeds
    torch.manual_seed(123)

    # create env
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # create NNs
    ac_net: ActorCriticNet = ActorCriticNet(4, 2).to(device)
    optimizer = optim.Adam(ac_net.parameters(), lr=LR)

    # main loop
    for i_episode in count():
        state, _ = env.reset()
        ep_reward = 0

        for t in count():
            action = select_action(state, ac_net, device)
            state, reward, terminated, truncated, _ = env.step(action)
            ac_net.rewards.append(reward)
            ep_reward += reward
            done: bool = terminated or truncated
            if done:
                break
        finish_episode(ac_net, optimizer, device, GAMMA, EPS)

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # print & plot
        episode_durations.append(ep_reward)
        running_rewards.append(running_reward)
        print(f'\rEpisode: {i_episode}, Last reward: {ep_reward}, Avr. reward: {running_reward}', end='')
        plot_episode_durations(ax[0], info={
            'episode_durations': episode_durations,
            'running_rewards': running_rewards,
        })
        plt.pause(plot_rate)
        if i_episode % 50 == 0 and i_episode > 0:
            run_CartPole(select_action, ac_net, device)


        # stop condition
        if running_reward > reward_threshold:
            print(f'Solved. Avr. reward is above {reward_threshold}.')
            break

    run_CartPole(select_action, ac_net, device, n_episodes=10)
    plt.show()


if __name__ == '__main__':
    main()