import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from globals import *
from collections import OrderedDict


def render_field(ax: matplotlib.axes.Axes, info):
    ax.cla()
    markersize = 10
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


def render_tiger_deer_field(ax: matplotlib.axes.Axes, info):
    ax.cla()
    markersize = 10
    env_name = info['env_name']
    tigers_dict = info['tigers_dict']
    deer_dict = info['deer_dict']
    input_field = info['field']
    iteration = info['iteration']
    n_episode = info['n_episode']
    width = input_field.shape[0]
    height = input_field.shape[1]
    alive_deer_dict = {d_name: d_params for d_name, d_params in deer_dict.items() if d_params['alive']}
    alive_tiger_dict = OrderedDict((t_name, t_params) for t_name, t_params in tigers_dict.items() if t_params['alive'])

    # field
    field_x, field_y = [], []
    for x in range(width):
        for y in range(height):
            if input_field[x, y] == 1:
                field_x.append(x)
                field_y.append(y)
    ax.plot(field_x, field_y, 's', c='gray', markersize=markersize)

    # tigers: moves and attacks
    tigers_x, tigers_y = [], []
    attack_lines = []
    for t_name, t_params in alive_tiger_dict.items():
        t_x, t_y = t_params['loc'][0], t_params['loc'][1]
        t_hp = f"{t_params['hp']: .2f}"
        tigers_x.append(t_x)
        tigers_y.append(t_y)
        attack_lines.append((t_params['loc'], t_params['attack']))
        ax.text(t_x, t_y + 1, t_hp, color='red', fontsize=8, ha='center', va='center')
    ax.plot(tigers_x, tigers_y, 'o', c='red', markersize=markersize)
    line_collection = LineCollection(attack_lines, colors='k', linewidths=1)
    ax.add_collection(line_collection)

    # deer
    deer_x, deer_y = [], []
    for d_name, d_params in alive_deer_dict.items():
        d_x, d_y = d_params['loc'][0], d_params['loc'][1]
        d_hp = f"{d_params['hp']: .2f}"
        deer_x.append(d_x)
        deer_y.append(d_y)
        ax.text(d_x, d_y + 1, d_hp, color='blue', fontsize=8, ha='center', va='center')
    ax.plot(deer_x, deer_y, 'o', c='blue', markersize=markersize)

    main_agent_params = tigers_dict['tiger_0']
    if main_agent_params['alive']:
        obs_radius = main_agent_params['obs_radius']
        rec_x = main_agent_params['loc'][0] - obs_radius
        rec_y = main_agent_params['loc'][1] - obs_radius
        rec_side = obs_radius * 2
        rect = plt.Rectangle((rec_x, rec_y), rec_side, rec_side, color='brown', alpha=0.07)
        ax.add_patch(rect)
        ax.plot(main_agent_params['loc'][0], main_agent_params['loc'][1], 'o', c='brown', markersize=markersize)

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
    ax.set_title(f'{env_name} | episode: {n_episode} | iter: {iteration}', fontweight="bold", size=20)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def render_td_agent_view(ax: matplotlib.axes.Axes, info: dict):
    ax.cla()
    markersize = 20
    env_name = info['env_name']
    tigers_dict = info['tigers_dict']
    deer_dict = info['deer_dict']
    input_field = info['field']
    iteration = info['iteration']
    n_episode = info['n_episode']
    field = info['field']
    out_obs = info['out_obs']
    width, height = field.shape[0], field.shape[1]

    # field
    ax.imshow(out_obs['tiger_0'][:, :, 4], origin='lower')
    # field_x, field_y = [], []
    # for x in range(width):
    #     for y in range(height):
    #         if field[x, y] == 1:
    #             field_x.append(x)
    #             field_y.append(y)
    # ax.plot(field_x, field_y, 's', c='gray', markersize=markersize)
    # agents
    # agents_x, agents_y = [], []
    # for x in range(width):
    #     for y in range(height):
    #         if agents_view[x, y] == 1:
    #             agents_x.append(x)
    #             agents_y.append(y)
    # ax.plot(agents_x, agents_y, 'o', c='blue', markersize=markersize)

    padding = 0.5
    # ax.set_xlim([-padding, width - 1 + padding])
    # ax.set_ylim([-padding, height - 1 + padding])
    # ax.set_ylim([0, 1 + 0.1])
    # ax.set_xticks(range(width))
    # ax.set_yticks(range(height))
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