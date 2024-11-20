import time

from magent2.environments import battle_v4, battlefield_v5, combined_arms_v6
from magent2.environments import tiger_deer_v3, tiger_deer_v4, gather_v5
from pettingzoo.utils import random_demo

render_mode='human'
# render_mode=None

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

env = tiger_deer_v4.env(
    map_size=105, minimap_mode=False, tiger_step_recover=-0.1, deer_attacked=-0.1, max_cycles=500, extra_features=False,
    render_mode=render_mode
)

env.reset()
done_dict = {a_name: False for a_name in env.agents}

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination:
        env.step(None)
        done_dict[agent] = True
        continue
    action = env.action_space(agent).sample()
    env.step(action)
env.close()