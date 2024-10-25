from magent2.environments import battle_v4, battlefield_v5, combined_arms_v6
from pettingzoo.utils import random_demo

render_mode='human'
# render_mode=None

# env = battle_v4.env(render_mode=render_mode)

# env = battlefield_v5.env(map_size=80, minimap_mode=False, step_reward=-0.005,
# dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
# max_cycles=1000, extra_features=False, render_mode=render_mode)

env = combined_arms_v6.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000,
extra_features=False, render_mode=render_mode)

random_demo(env, render=True, episodes=1)