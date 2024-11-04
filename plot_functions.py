from globals import *


def plot_episode_durations(ax, info):
    ax.cla()
    episode_durations = info['episode_durations']
    running_rewards = info['running_rewards']
    env_name = info['env_name']
    ax.plot(episode_durations)
    ax.plot(running_rewards, color='r')

    # ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    # ax.set_xlim([min(n_agents_list), max(n_agents_list)])
    # ax.set_ylim([0, 1 + 0.1])
    # ax.set_xticks(n_agents_list)
    # ax.set_xlabel('N agents', fontsize=27)
    # ax.set_ylabel('Success Rate', fontsize=27)
    # # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    ax.set_title(f'{env_name}', fontweight="bold", size=20)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def plot_action_std(ax, info):
    ax.cla()
    action_std_list = info['action_std_list']
    env_name = info['env_name']
    ax.plot(action_std_list)

    # ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    # ax.set_xlim([min(n_agents_list), max(n_agents_list)])
    # ax.set_ylim([0, 1 + 0.1])
    # ax.set_xticks(n_agents_list)
    # ax.set_xlabel('N agents', fontsize=27)
    # ax.set_ylabel('Success Rate', fontsize=27)
    # # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    ax.set_title(f'{env_name}', fontweight="bold", size=20)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()