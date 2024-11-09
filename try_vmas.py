import vmas
import time

# Create the environment
env = vmas.make_env(
    scenario="waterfall", # can be scenario name or BaseScenario class
    num_envs=1,
    device="cpu", # Or "cuda" for GPU
    continuous_actions=True,
    max_steps=None, # Defines the horizon. None is infinite horizon.
    seed=None, # Seed of the environment
    n_agents=3  # Additional arguments you want to pass to the scenario
)
# Reset it
obs = env.reset()

env.render(
    mode="human",  # "rgb_array" returns image, "human" renders in display
    agent_index_focus=0,  # If None keep all agents in camera, else focus camera on specific agent
)
# Step it with deterministic actions (all agents take their maximum range action)
for i in range(1000):
    obs, rews, dones, info = env.step(env.get_random_actions())
    time.sleep(0.1)
    print(f'{i}')