import matplotlib
from circle import *

matplotlib.use('TkAgg')


def computeEigenvector(M, W):
	A = np.transpose(M) @ np.transpose(W) @ W @ M
	eigValues, eigVectors = np.linalg.eig(A)
	return eigVectors[:, [np.argmin(eigValues)]]


def updateCircle(C, v):
	v = [v[i][0] for i in range(4)]
	C.center.x = updateCx(v[0], v[1])
	C.center.y = updateCy(v[0], v[2])
	C.r = updateRadius(v[0], v[1], v[2], v[3])


def updateCx(v1, v2):
	return -v2 / (2 * v1)


def updateCy(v1, v3):
	return -v3 / (2 * v1)


def updateRadius(v1, v2, v3, v4):
	return np.sqrt((v2 ** 2 + v3 ** 2) / (4 * (v1 ** 2)) - v4 / v1)
