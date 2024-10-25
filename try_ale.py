# import gymnasium
# import ale_py
#
# gymnasium.register_envs(ale_py)
#
# env = gymnasium.make("ALE/Pong-v5", render_mode="human")
# env.reset()
# for _ in range(100):
#     action = env.action_space.sample()
#
#     obs, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         obs, info = env.reset()
#
# env.close()

from pettingzoo.atari import space_invaders_v2
env = space_invaders_v2.env(render_mode="human")

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)
env.close()