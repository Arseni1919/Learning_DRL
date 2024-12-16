# from globals import *
#
# means = torch.zeros(400, 1)
# cov_mat = torch.eye(1).squeeze()
# m = MultivariateNormal(means, cov_mat)
# print(m.sample())  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`

# radius = 10
# for i in range(-radius, radius+1):
#     print(i)


l1 = [1, 2, 3]
l2 = [4, 5, 6]

print(l1 + l2)

import random
for i in range(10):
    print(random.randint(0, 9))