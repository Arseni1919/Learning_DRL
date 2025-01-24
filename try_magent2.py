

from magent2.environments import battle_v4, battlefield_v5, combined_arms_v6
from magent2.environments import tiger_deer_v3, tiger_deer_v4, gather_v5
from pettingzoo.utils import random_demo



# from magent2.environments.adversarial_pursuit import parallel_env
# from magent2.environments.battle import parallel_env
# from magent2.environments.battlefield import parallel_env
# from magent2.environments.combined_arms import parallel_env
from magent2.environments.gather import parallel_env
# from magent2.environments.tiger_deer import parallel_env
import time

render_mode='human'
# render_mode=None

# for tiger_deer end to change n of agents -> change map_size
# env = parallel_env(map_size=10, render_mode=render_mode, max_cycles=200, minimap_mode=False)

env = parallel_env(render_mode=render_mode, max_cycles=200, minimap_mode=False)
observations, infos = env.reset()

i_step = 0
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    i_step += 1
    print(f'{i_step}')
    time.sleep(0.01)
env.close()


# env = battlefield_v5.env(
#     map_size=50, minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1,
#     attack_opponent_reward=0.2, max_cycles=1000, extra_features=False, render_mode=render_mode
# )

# env = combined_arms_v6.env(
#     map_size=26, minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1,
#     attack_opponent_reward=0.2, max_cycles=1000, extra_features=False, render_mode=render_mode
# )

# env = tiger_deer_v4.env(
#     map_size=25, minimap_mode=False, tiger_step_recover=-0.1, deer_attacked=-0.1, max_cycles=500, extra_features=False,
#     render_mode=render_mode
# )

# env = gather_v5.env(
#     minimap_mode=False, step_reward=-0.01, attack_penalty=-0.1, dead_penalty=-1, attack_food_reward=0.5,
#     max_cycles=500, extra_features=False, render_mode=render_mode
# )


# random_demo(env, render=True, episodes=10)

# env = tiger_deer_v4.env(
#     map_size=105, minimap_mode=False, tiger_step_recover=-0.1, deer_attacked=-0.1, max_cycles=500, extra_features=False,
#     render_mode=render_mode
# )
#
# env.reset()
# done_dict = {a_name: False for a_name in env.agents}
#
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     if termination:
#         env.step(None)
#         done_dict[agent] = True
#         continue
#     action = env.action_space(agent).sample()
#     env.step(action)
# env.close()


# env = basketball_pong_v3.parallel_env(render_mode="human")