import matplotlib.axes
import matplotlib.pyplot as plt
from torch.fx.experimental.unification.unification_tools import first

from globals import *


def render_field(ax: matplotlib.axes.Axes, info):
    ax.cla()
    markersize = 20
    env_name = info['env_name']
    target = info['target']
    agents_loc = info['agents_loc']
    input_field = info['field']
    iteration = info['iteration']
    width = input_field.shape[0]
    height = input_field.shape[1]
    # field
    field_x, field_y = [], []
    for x in range(width):
        for y in range(height):
            if input_field[x, y] == 1:
                field_x.append(x)
                field_y.append(y)
    ax.plot(field_x, field_y, 's', c='gray', markersize=markersize)
    # target
    ax.plot(target[0], target[1], '*', c='red', markersize=markersize)
    # agents
    agents_x, agents_y = [], []
    for agent_name, loc in agents_loc.items():
        agents_x.append(loc[0])
        agents_y.append(loc[1])
    ax.plot(agents_x, agents_y, 'o', c='blue', markersize=markersize)

    main_agent_loc = agents_loc['agent_0']
    ax.plot(main_agent_loc[0], main_agent_loc[1], 'o', c='purple', markersize=markersize)

    padding = 0.5
    ax.set_xlim([-padding, width + padding])
    ax.set_ylim([-padding, height + padding])
    # ax.set_ylim([0, 1 + 0.1])
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    # ax.set_xlabel('N agents', fontsize=27)
    # ax.set_ylabel('Success Rate', fontsize=27)
    # # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    ax.set_title(f'{env_name} | iteration: {iteration}', fontweight="bold", size=20)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def render_agent_view(ax: matplotlib.axes.Axes, info: dict):
    ax.cla()
    markersize = 20
    agent_obs = info['agent_obs']
    map_view = agent_obs[0]
    agents_view = agent_obs[1]
    target_view = agent_obs[2]
    width, height = map_view.shape[0], map_view.shape[1]
    # field
    field_x, field_y = [], []
    for x in range(width):
        for y in range(height):
            if map_view[x, y] == 1:
                field_x.append(x)
                field_y.append(y)
    ax.plot(field_x, field_y, 's', c='gray', markersize=markersize)
    # agents
    agents_x, agents_y = [], []
    for x in range(width):
        for y in range(height):
            if agents_view[x, y] == 1:
                agents_x.append(x)
                agents_y.append(y)
    ax.plot(agents_x, agents_y, 'o', c='blue', markersize=markersize)
    # target
    target_x, target_y = [], []
    for x in range(width):
        for y in range(height):
            if target_view[x, y] == 1:
                target_x.append(x)
                target_y.append(y)
    ax.plot(target_x, target_y, '*', c='red', markersize=markersize)

    padding = 0.5
    ax.set_xlim([-padding, width - 1 + padding])
    ax.set_ylim([-padding, height - 1 + padding])
    # ax.set_ylim([0, 1 + 0.1])
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    # ax.set_xlabel('N agents', fontsize=27)
    # ax.set_ylabel('Success Rate', fontsize=27)
    # # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    ax.set_title(f"Agent's View", fontweight="bold", size=20)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()