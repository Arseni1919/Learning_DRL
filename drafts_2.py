# import numpy as np
#
# # Create a 5x5 matrix
# matrix = np.array([[ 1,  2,  3,  4,  5],
#                    [ 6,  7,  8,  9, 10],
#                    [11, 12, 13, 14, 15],
#                    [16, 17, 18, 19, 20],
#                    [21, 22, 23, 24, 25]])
#
# # Select a sub-matrix (2x3), starting from row index 1 and column index 2
# radius = 3
# i_x = 2
# i_y = 3
# sub_matrix = matrix[i_x:i_x + radius, i_y:i_y + radius]
#
# print("Original Matrix:")
# print(matrix)
# print("\nSub-matrix:")
# print(sub_matrix)

import numpy as np
import matplotlib.pyplot as plt

# Define the probability ratio range
r_t = np.linspace(0.5, 1.5, 100)  # Covers both underestimation and overestimation

# Define advantage values
A_positive = 1.0  # Positive advantage (good action)
A_negative = -1.0  # Negative advantage (bad action)

# Define PPO clipping threshold
epsilon = 0.2

# Compute objectives for positive advantage
unclipped_positive = r_t * A_positive
clipped_positive = np.clip(r_t, 1 - epsilon, 1 + epsilon) * A_positive
ppo_objective_positive = np.minimum(unclipped_positive, clipped_positive)

# Compute objectives for negative advantage
unclipped_negative = r_t * A_negative
clipped_negative = np.clip(r_t, 1 - epsilon, 1 + epsilon) * A_negative
ppo_objective_negative = np.minimum(unclipped_negative, clipped_negative)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot for positive advantage
ax[0].plot(r_t, unclipped_positive, label="Unclipped", linestyle="--", color="blue")
ax[0].plot(r_t, clipped_positive, label="Clipped", linestyle="--", color="red")
ax[0].plot(r_t, ppo_objective_positive, label="Final PPO Loss", color="black")
ax[0].axvline(1 - epsilon, color="gray", linestyle=":", label="Clipping Region")
ax[0].axvline(1 + epsilon, color="gray", linestyle=":")
ax[0].set_title("Positive Advantage (Good Action)")
ax[0].set_xlabel("Probability Ratio (r_t)")
ax[0].set_ylabel("Loss Value")
ax[0].legend()
ax[0].grid(True)

# Plot for negative advantage
ax[1].plot(r_t, unclipped_negative, label="Unclipped", linestyle="--", color="blue")
ax[1].plot(r_t, clipped_negative, label="Clipped", linestyle="--", color="red")
ax[1].plot(r_t, ppo_objective_negative, label="Final PPO Loss", color="black")
ax[1].axvline(1 - epsilon, color="gray", linestyle=":", label="Clipping Region")
ax[1].axvline(1 + epsilon, color="gray", linestyle=":")
ax[1].set_title("Negative Advantage (Bad Action)")
ax[1].set_xlabel("Probability Ratio (r_t)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()