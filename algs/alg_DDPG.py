from alg_DDPG_sup import *


def main():
    # parameters
    hidden_dim = 400
    batch_size = 256
    buffer_size = 200000
    actor_lr = 5e-4
    critic_lr = 1e-3
    tau = 5e-3
    gamma = 0.995
    noise_scale = 0.5
    noise_decay = 0.998
    n_episodes = 200
    max_steps = 100

    running_reward = 35
    reward_threshold = 930
    # render_mode = "human"  # "human" / "rgb_array" / None
    render_mode = None
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    # for plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.001
    episode_durations: List[int] = []
    running_rewards: List[float] = []

    # seeds
    torch.manual_seed(123)

    # create env
    env, env_name = gym.make('InvertedDoublePendulum-v5', reset_noise_scale=0.1, render_mode=render_mode), 'InvertedDoublePendulum-v5'
    # env, env_name = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, render_mode=render_mode), 'HalfCheetah-v5'

    # create NNs
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # DDPG: state_dim, action_dim, hidden_dim, buffer_size, batch_size, actor_lr, critic_lr, tau, gamma
    agent = DDPG(state_dim, action_dim, hidden_dim, buffer_size, batch_size, actor_lr, critic_lr, tau, gamma, device)

    # main loop
    for i_episode in count():
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            action = agent.act(state, noise_scale)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done: bool = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            noise_scale *= noise_decay
            state = next_state
            if done:
                break
            # print
            print(f'\rEpisode: {i_episode}, step: {t}, Last reward: {ep_reward}, Running reward: {running_reward}', end='')

        running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward

        # print & plot
        episode_durations.append(ep_reward)
        running_rewards.append(running_reward)
        # print(f'\rEpisode: {i_episode}, Last reward: {ep_reward}, Avr. reward: {running_reward}', end='')
        plot_episode_durations(ax[0], info={
            'episode_durations': episode_durations,
            'running_rewards': running_rewards,
        })
        plt.pause(plot_rate)
        if i_episode % 50 == 0 and i_episode > 0:
            run_mujoco(env_name, agent, noise_scale)

        # stop condition
        if running_reward > reward_threshold:
            print(f'\n\nSolved. Avr. reward is above {reward_threshold}.')
            break

    run_mujoco(env_name, agent, noise_scale, n_episodes=10)
    plt.show()


if __name__ == '__main__':
    main()