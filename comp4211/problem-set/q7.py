import numpy as np
from math import *

points = [(1, 0), (2, 1), (8, 8), (9, 6), (9, 8)]
dist = lambda x, y: sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def closest_points(points):
    distances = []
    for i, x in enumerate(points):
        for j in range(i + 1, len(points)):
            distances.append((x, points[j], dist(x, points[j])))

    return sorted(distances, key=lambda t: t[2])



d = closest_points(points)
print(*d, sep='\n')

