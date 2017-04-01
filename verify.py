#!/usr/bin/env python3

from math import *

ref1 = [ 0.0, 0.0, 0.0]
ref2 = [ 100.0, 0.0, 0.0]
ref3 = [ 150.0, 200.0, 0.0]

def sqdist(p1, p2):
    dist = 0
    for i in range(0,3):
        dist += (p1[i] - p2[i])**2
    return dist


def tri(point, refs):
    d = refs[1][0]
    i = refs[2][0]
    j = refs[2][1]


    ds = [0, 0, 0]
    for t in range(0,3):
        ds[t] = sqdist(point, refs[t])

    x = ( ds[0] - ds[1] + d**2) / (2 * d)
    y = (( ds[0] - ds[2] + i**2 + j**2)  / (2 * j)) - ( i * x / j)
    z = sqrt( abs(ds[0] - x**2 - y**2))

    return [x, y, z] 


if __name__ == '__main__':
    p = [49.4404, 49.7267, 50.1733]
    print(tri(p, [ref1, ref2, ref3]))
