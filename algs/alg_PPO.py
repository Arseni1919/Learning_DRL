from alg_PPO_sup import *


def main():
    # parameters
    hidden_dim = 400
    max_steps = 100
    update_time_interval = max_steps * 4
    k_epochs = 40
    eps_clip = 0.2
    gamma = 0.995
    actor_lr = 5e-4
    critic_lr = 1e-3
    action_std_init = 0.6
    action_std_decay_rate = 0.01
    min_action_std = 0.1
    # action_std_decay_freq = int(2.5e5)
    action_std_decay_freq = int(500)

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
    action_std_list: List[float] = []

    # seeds
    torch.manual_seed(123)

    # create env
    env, env_name = gym.make('InvertedDoublePendulum-v5', reset_noise_scale=0.1, render_mode=render_mode), 'InvertedDoublePendulum-v5'
    # env, env_name = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, render_mode=render_mode), 'HalfCheetah-v5'

    # create NNs
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(
        state_dim,
        action_dim,
        hidden_dim,
        update_time_interval,
        action_std_init,
        actor_lr,
        critic_lr,
        k_epochs,
        gamma,
        eps_clip,
        device
    )

    # main loop
    time_step = 0
    for i_episode in count():
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            time_step += 1

            action, logprob, state_value = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            agent.store_transition(state, action, logprob, reward, state_value, done)
            ep_reward += reward

            # update PPO agent
            if time_step % update_time_interval == 0:
                agent.learn()

            # if continuous action space -> decay action std of output action distribution
            if time_step % action_std_decay_freq == 0:
            # if time_step % update_time_interval * 10 == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            state = next_state
            if done:
                break
            # print
            print(f'\rEpisode: {i_episode}, step: {t}, Last reward: {ep_reward}, Running reward: {running_reward}', end='')

        running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward

        # print & plot
        episode_durations.append(ep_reward)
        running_rewards.append(running_reward)
        action_std_list.append(agent.action_std)
        # print(f'\rEpisode: {i_episode}, Last reward: {ep_reward}, Avr. reward: {running_reward}', end='')
        if to_plot:
            plot_episode_durations(ax[0], info={
                'episode_durations': episode_durations,
                'running_rewards': running_rewards,
                'env_name': env_name,
            })
            plot_action_std(ax[1], info={
                'action_std_list': action_std_list,
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