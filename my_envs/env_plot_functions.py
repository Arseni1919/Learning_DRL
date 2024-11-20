import matplotlib.axes
import matplotlib.pyplot as plt
from torch.fx.experimental.unification.unification_tools import first

from globals import *


def render_field(ax: matplotlib.axes.Axes, info):
    ax.cla()
    env_name = info['env_name']
    target = info['target']
    agents_loc = info['agents_loc']
    field = info['field']
    # target
    field[target[0], target[1]] = 2
    for agent_name, loc in agents_loc.items():
        field[loc[0], loc[1]] = 1
    ax.imshow(field, cmap='gray')

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