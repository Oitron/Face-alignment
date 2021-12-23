import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#show an image and its landmarks
def showData_1(img, landmarks1):
    landmarks1T = landmarks1.copy().T
    x1 = landmarks1T[0]
    y1 = landmarks1T[1]
    plt.imshow(img, cmap='gray')
    plt.plot(x1, y1, 'ro',  markersize=1.5)
    plt.show()

def showData_2(img, landmarks1, landmarks2):
    landmarks1T = landmarks1.copy().T
    x1 = landmarks1T[0]
    y1 = landmarks1T[1]
    landmarks2T = landmarks2.copy().T
    x2 = landmarks2T[0]
    y2 = landmarks2T[1]
    plt.imshow(img, cmap='gray')
    plt.plot(x1, y1, 'ro', markersize=1.5)
    plt.plot(x2, y2, 'bo', markersize=1.5)
    plt.show()

def showBoundingBox(img, landmarks, topLeft, bottomRight):
    width = bottomRight[0] - topLeft[0] + 1
    height = bottomRight[1] - topLeft[1] + 1
    landmarksT = landmarks.copy().T
    x = landmarksT[0]
    y = landmarksT[1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(img, cmap='gray')
    rect = plt.Rectangle(topLeft, width, height, fill=False, edgeColor='blue', linewidth=1)
    plt.plot(x, y, 'ro', markersize=1.5)
    ax.add_patch(rect)
    plt.show()

def display_10_1(imgs_set, lmks_set):
    indices = [np.random.randint(0, len(imgs_set)) for i in range(10)]
    for index in indices:
        showData_1(imgs_set[index], lmks_set[index])

def display_10_2(imgs_set, lmks_set1, lmks_set2):
    indices = [np.random.randint(0, len(imgs_set)) for i in range(10)]
    for index in indices:
        showData_2(imgs_set[index], lmks_set1[index], lmks_set2[index])

def repeatLmk(lmk, size):
        lmk = np.repeat(lmk[np.newaxis,:,:], size, axis=0)
        return lmk

# (168, 2) -> 136
def reshapeLmkCoords(lmks):
    lmks = lmks.reshape((lmks.shape[0], lmks.shape[1] * lmks.shape[2]))
    return lmks

# 136 -> (168, 2)
def recoverLmkCoords(lmks_set):
    return lmks_set.reshape((lmks_set.shape[0], int(lmks_set.shape[1]/2), 2))

def calculateError(groundTruth_lmks_set, predict_lmks_set):
    error = 0
    for i in range(len(groundTruth_lmks_set)):
        error += mean_squared_error(groundTruth_lmks_set[i], predict_lmks_set[i])
    return error