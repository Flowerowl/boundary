# encoding:utf-8
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib.patches import Polygon


def chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def draw_boundary(points):
    hull = ConvexHull(points)
    plt.plot(points[:,0], points[:,1], 'o')
    cent = np.mean(points, 0)
    pts = []
    for pt in points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())

    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1], p[0] - cent[0]))
    pts = pts[0::2]
    pts.insert(len(pts), pts[0])

    pts = chaikins_corner_cutting(pts)
    hull = ConvexHull(pts)

    poly = Polygon(1.5*(np.array(pts)-cent)+cent, facecolor='green', alpha=0.2)
    poly.set_capstyle('round')
    plt.gca().add_patch(poly)


if __name__ == "__main__":
    X = []
    with open("points.csv","rb") as f:
        reader = csv.reader(f)
        for line in reader:
            X.append(line)
        X = np.array(X, np.float)

    db = DBSCAN(eps=50, min_samples=5, metric='euclidean').fit(X)
    labels = db.labels_

    print(set(labels))
    unique_labels = set(labels)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]

    for label, col in zip(unique_labels, colors):
        if label == -1:
            continue
        indexs = [i for i, x in enumerate(labels) if x == label]
        plt.plot(X[indexs][:, 0], X[indexs][:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=4)
        draw_boundary(X[indexs])

    plt.show()
