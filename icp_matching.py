#!/usr/bin/env python
#coding: utf8

'''
参考文献
[1] "ICPアルゴリズムを利用したSLAM用MATLABサンプルプログラム - MyEnigma",
    http://myenigma.hatenablog.com/entry/20140617/1402971928 ,
    2016年11月10日アクセス.
'''

import csv
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def read_file(path):
    ax = []
    ay = []
    with open(path) as f:
        for line in csv.reader(f):
            if len(line) == 0:
                break

            x = int(line[0])
            y = int(line[1])
            ax.append(x)
            ay.append(y)
    return np.array((ax, ay))

def calc_closest_points(data1, data2):
    nrnb = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
    nrnb.fit(data1.T)
    distances, indices = nrnb.kneighbors(data2.T)
    dt = distances.T[0]
    it = indices.T[0]

    distance_mean = np.sum(dt) / len(dt)
    closests = np.array([data1[0, it],
                         data1[1, it]])
    return closests, distance_mean

def calc_by_svd(data, closests):
    M = data
    S = closests
    mm = np.mean(M, axis=1)
    ms = np.mean(S, axis=1)

    Ms = np.array((M[0] - mm[0], M[1] - mm[1]))
    Ss = np.array((S[0] - ms[0], S[1] - ms[1]))
    W = np.dot(Ss, Ms.T)
    U, _, V = np.linalg.svd(W)

    R = np.dot(U, V.T)
    t = mm - np.dot(R, ms)
    return R, t

def icp(data1, data2, max_iterates=100, error_threshold=0.00001):
    rR = np.identity(2)
    rt = np.array((0, 0))

    errors = []
    pre = float('inf')
    for i in range(max_iterates):
        closests, distance_mean = calc_closest_points(data1, data2)
        R, t = calc_by_svd(data2, closests)

        data2 = np.dot(R, data2)
        data2 = np.array((data2[0] - t[0], data2[1] - t[1]))

        rR = np.dot(R, rR)
        rt = np.dot(R, rt) - t

        error = np.abs(pre - distance_mean)
        errors.append(error)
        if error < error_threshold:
            break
        pre = distance_mean

    return rR, rt, data2, errors

def main():
    name1 = sys.argv[1]
    name2 = sys.argv[2]
    print name1, name2
    out1 = read_file(name1)
    out2 = read_file(name2)

    R, t, tr, errors = icp(out1, out2)

    print 'R', R
    print 't', t

    plt.subplot(121)
    plt.scatter(out1[0], out1[1], marker='o', color='red')
    plt.scatter(out2[0], out2[1], marker='o', color='green')
    plt.scatter(tr[0], tr[1], marker='x', color='blue')
    plt.title('%s - %s' % (name1, name2))
    plt.legend((name1, name2, 'moved %s' % name2),
               loc='upper left')

    plt.subplot(122)
    plt.plot(errors)
    plt.title('error')

    plt.show()

if __name__ == '__main__':
    main()
