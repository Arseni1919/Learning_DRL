from globals import *
from smac.env import StarCraft2Env


def run_CartPole(select_action, ac_net, device, render_mode: str = 'human', n_episodes: int = 1):
    env = gym.make("CartPole-v1", render_mode=render_mode)

    for i_episode in range(n_episodes):
        state, _ = env.reset()

        for t in count():
            action = select_action(state, ac_net, device)
            state, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            if done:
                break


def run_mujoco(env_name: str, agent, noise_scale: float = 0.1, render_mode: str = 'human', n_episodes: int = 1):
    if env_name == 'HalfCheetah-v5':
        env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, render_mode=render_mode)
    elif env_name == 'InvertedDoublePendulum-v5':
        env = gym.make('InvertedDoublePendulum-v5', reset_noise_scale=0.1, render_mode=render_mode)
    else:
        return
    # Reset the environment to get the initial state
    obs, info = env.reset()

    for i_episode in range(n_episodes):

        for _ in range(100):  # Run for 1000 time steps
            # action = env.action_space.sample()  # random action
            action = agent.get_pure_action(obs, noise_scale)

            # Step the environment forward and get results
            obs, reward, done, truncated, info = env.step(action)

            # time.sleep(1)

            # Check if the episode is done
            if done or truncated:
                # obs, info = env.reset()
                break

    # Close the environment properly
    env.close()


def run_box2d(env_name: str, agent, noise_scale: float = 0.1, render_mode: str = 'human', n_episodes: int = 1):
    if env_name == 'BipedalWalker-v3':
        env, env_name = gym.make("BipedalWalker-v3", hardcore=True, render_mode=render_mode), "BipedalWalker-v3"
    else:
        return
    # Reset the environment to get the initial state
    obs, info = env.reset()

    for i_episode in range(n_episodes):

        for _ in range(100):  # Run for 1000 time steps
            # action = env.action_space.sample()  # random action
            action = agent.get_pure_action(obs, noise_scale)

            # Step the environment forward and get results
            obs, reward, done, truncated, info = env.step(action)

            # time.sleep(1)

            # Check if the episode is done
            if done or truncated:
                # obs, info = env.reset()
                break

    # Close the environment properly
    env.close()


def run_smac(map_name: str, alg: Any) -> None:
    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = alg.get_action(avail_actions, avail_actions_ind, obs[agent_id])
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print(f"Total reward in episode {e} = {episode_reward}")

    # env.save_replay()
    env.close()

