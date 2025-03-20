import numpy as np
import matplotlib.pyplot as plt

# Define advantage values
A_positive = 1.0  # Positive advantage (good action)
A_negative = 1.0  # Negative advantage (bad action)

# Define PPO clipping threshold
epsilon = 0.2

r_t = 0.6
# Compute objectives for negative advantage
unclipped_negative = r_t * A_negative
clipped_negative = np.clip(r_t, 1 - epsilon, 1 + epsilon) * A_negative
ppo_objective_negative = np.minimum(unclipped_negative, clipped_negative)

print(ppo_objective_negative)




