from globals import *

means = torch.zeros(400, 1)
cov_mat = torch.eye(1).squeeze()
m = MultivariateNormal(means, cov_mat)
print(m.sample())  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
