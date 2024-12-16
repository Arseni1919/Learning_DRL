import numpy as np

# Create a 5x5 matrix
matrix = np.array([[ 1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# Select a sub-matrix (2x3), starting from row index 1 and column index 2
radius = 3
i_x = 2
i_y = 3
sub_matrix = matrix[i_x:i_x + radius, i_y:i_y + radius]

print("Original Matrix:")
print(matrix)
print("\nSub-matrix:")
print(sub_matrix)