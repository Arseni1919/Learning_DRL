from alg_SAC_sup import *


def main():
    # parameters
    hidden_dim = 512
    max_steps = 100
    actor_lr = 3e-4
    value_lr = 3e-4
    soft_q_lr = 3e-4
    gamma = 0.995
    soft_tau = 1e-2
    buffer_size = 100000
    batch_size = 128

    running_reward = 35
    reward_threshold = 930
    # render_mode = "human"  # "human" / "rgb_array" / None
    render_mode = None
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"

    # for plots
    to_plot = True
    # to_plot = False
    if to_plot:
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
    agent = SAC(
        state_dim,
        action_dim,
        hidden_dim,
        actor_lr,
        value_lr,
        soft_q_lr,
        gamma,
        soft_tau,
        buffer_size,
        batch_size,
        device
    )

    # main loop
    time_step = 0
    for i_episode in count():
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            time_step += 1

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            ep_reward += reward

            # update SAC agent
            agent.learn()

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
        if to_plot:
            plot_episode_durations(ax[0], info={
                'episode_durations': episode_durations,
                'running_rewards': running_rewards,
                'env_name': env_name,
            })
            plt.pause(plot_rate)
            if i_episode % 200 == 0 and i_episode > 0:
                run_mujoco(env_name, agent)

        # stop condition
        if running_reward > reward_threshold:
            print(f'\n\nSolved. Avr. reward is above {reward_threshold}.')
            break

    run_mujoco(env_name, agent, n_episodes=10)
    plt.show()


if __name__ == '__main__':
    main()