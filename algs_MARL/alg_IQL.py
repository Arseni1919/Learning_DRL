import matplotlib.pyplot as plt

from alg_IQL_sup import *
from plot_functions import *
from global_functions import *


def main():
    # parameters
    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: float = 10000
    TAU: float = 0.005  # update rate of the target network
    LR: float = 1e-5  # learning rate of the AdamW optimizer
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else
    #     "mps" if torch.backends.mps.is_available() else
    #     "cpu"
    # )
    device = 'cpu'

    # for plots
    to_plot = True
    # to_plot = False
    running_reward = 0
    running_rewards: List[int] = []
    ep_rewards: List[int] = []
    if to_plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.001

    #seeds
    torch.manual_seed(123)

    # create env
    map_name = "3m"
    env = StarCraft2Env(map_name=map_name)

    # parameters after env init
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    obs_shape = env_info['obs_shape']

    # create alg
    alg: AlgIQL = AlgIQL(
        n_agents, n_actions, obs_shape,
        LR, EPS_START, EPS_DECAY, EPS_END, TAU, BATCH_SIZE, GAMMA,
        device
    )

    # main loop
    n_step = 0
    for i_episode in count():
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            n_step += 1
            obs = env.get_obs()
            # state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = alg.select_actions(obs, n_step, env)  # alg
            reward, terminated, _ = env.step(actions)
            new_obs = env.get_obs()
            alg.add_to_memories(obs, actions, new_obs, reward, terminated)  # alg
            alg.learn()  # alg


            episode_reward += reward

        # print(f"Total reward in episode {i_episode} = {episode_reward}")
        running_reward = 0.01 * episode_reward + (1 - 0.01) * running_reward
        running_rewards.append(running_reward)
        ep_rewards.append(episode_reward)
        if to_plot:
            plot_running_rewards(ax[0], info={
                'env_name': map_name, 'running_rewards': running_rewards, 'ep_rewards': ep_rewards
            })
            plt.pause(plot_rate)

    print('\n--- Completed ---')
    env.close()
    plt.show()


if __name__ == '__main__':
    main()