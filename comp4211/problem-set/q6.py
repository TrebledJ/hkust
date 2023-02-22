import numpy as np
from math import *


# Partitional Clustering
X = [(0, 5), (1, 3), (2, 4), (6, 2), (7, 0), (8, 3), (9, 1)]
dist = lambda x, y: sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def median(xs):
    xs = np.array(xs)
    n = xs.shape[0]
    if n % 2:
        return tuple(xs[n // 2, :])
    else:
        return tuple((xs[n // 2 - 1, :] + xs[n // 2, :]) / 2)


m = [(4, 2), (11, 3)]

n = 5
for i in range(n):
    clusters = [[] for _ in range(len(m))]
    for j, x in enumerate(X):
        dm = [dist(mx, x) for mx in m]
        
        if dm[0] < dm[1]:
            bold1, bold2 = r'\textbf', ''
            c = 0
        else:
            bold1, bold2 = '', r'\textbf'
            c = 1

        clusters[c].append(x)
        print(f'\item $p_{j+1}$ {bold1}{{distance to {m[0]}: {dm[0]:.3f}}}; {bold2}{{distance to {m[1]}: {dm[1]:.6f}}}')

    changed = False
    for c, cl in enumerate(clusters):
        print(f'$C_{c}$ points: \{{{", ".join(map(str, cl))}\}}')
        # med = tuple([median(x) for x in cl])
        med = median(cl)
        print(f'$C_{c}$ median: {med}')
        if med != m[c]:
            changed = True
            m[c] = med

    if not changed:
        print("No medians changed.")
        print("Algorithm finished after", i, "iterations.")
        print("Medians:")
        print(m)
        break

    print()