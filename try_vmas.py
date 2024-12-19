import time

import vmas

# Create the environment
env = vmas.make_env(
    # scenario="waterfall", # can be scenario name or BaseScenario class
    # scenario="dropout",
    # scenario="transport",
    # scenario="wheel",
    # scenario="drone",
    # scenario="kinematic_bicycle",
    # scenario="road_traffic",
    # scenario="multi_give_way",
    # scenario="football",
    # scenario="give_way",
    # scenario="simple",
    scenario="simple_adversary",
    num_envs=32,
    device="cpu", # Or "cuda" for GPU
    continuous_actions=True,
    max_steps=None, # Defines the horizon. None is infinite horizon.
    seed=None, # Seed of the environment
    n_agents=3  # Additional arguments you want to pass to the scenario
)
# Reset itr
obs = env.reset()

# Step it with deterministic actions (all agents take their maximum range action)
for i in range(1000):
    obs, rews, dones, info = env.step(env.get_random_actions())
    print(i)
    env.render(
        # mode="rgb_array",  # "rgb_array" returns image, "human" renders in display
        mode="human",  # "rgb_array" returns image, "human" renders in display
        # agent_index_focus=4, # If None keep all agents in camera, else focus camera on specific agent
        # index=0, # Index of batched environment to render
        # visualize_when_rgb=True,  # Also run human visualization when mode=="rgb_array"
    )

# import vmas
# import time
#
# # Create the environment
# env = vmas.make_env(
#     scenario="waterfall", # can be scenario name or BaseScenario class
#     num_envs=1,
#     device="cpu", # Or "cuda" for GPU
#     continuous_actions=True,
#     max_steps=None, # Defines the horizon. None is infinite horizon.
#     seed=None, # Seed of the environment
#     n_agents=3  # Additional arguments you want to pass to the scenario
# )
# # Reset it
# obs = env.reset()
#
# env.render(
#     mode="human",  # "rgb_array" returns image, "human" renders in display
#     agent_index_focus=0,  # If None keep all agents in camera, else focus camera on specific agent
# )
# # Step it with deterministic actions (all agents take their maximum range action)
# for i in range(1000):
#     obs, rews, dones, info = env.step(env.get_random_actions())
#     time.sleep(0.1)
#     print(f'{i}')