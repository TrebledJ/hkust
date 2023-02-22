import numpy as np
from math import *

S = np.array([(2, 1), (3, 5), (4, 3), (5, 6), (6, 7), (7, 8)])
mu = S.mean(axis=0)

print('covariance:')
cov = np.cov(S - mu, rowvar=False, bias=True)
print(cov)

print('eigenvalues:')
w, _ = np.linalg.eig(cov)
print(w)