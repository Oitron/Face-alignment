import os
import shutil  

import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

from Tools import repeatLmk, reshapeLmkCoords, recoverLmkCoords, calculateError, showData_2


class SDM:
    def _init_(self):
        self.Rs = []
        self.As = []
        self.Es_tr = []
        self.predictLmks = []
        self.patchSize = 20
        
    def featureExtraction(self, img, lmk, sift):
        keypoints = [cv2.KeyPoint(point[0], point[1], size=self.patchSize) for point in lmk]
        _, features = sift.compute(img, keypoints)
        return features.flatten()

    def constructX(self, imgs, lmks, sift):
        X = []
        assert len(imgs) == imgs.shape[0]
        for i in range(len(imgs)):
            X.append(self.featureExtraction(imgs[i], lmks[i], sift))
        return np.array(X).T

    def processPCA(self, X):
        X_r = X.copy().T 
        pca = PCA(n_components=0.98, svd_solver='full')
        X_r = pca.fit_transform(X_r)
        A = pca.components_
        return X_r.T, A

    def computeR(self, X_r_i, delta_s):
        reg = LinearRegression(fit_intercept=False).fit(X_r_i.T, delta_s)
        R = reg.coef_
        return R

    def oneIterAlignment(self, imgs_set, lmks_gt, lmks_in):
        sift = cv2.SIFT_create()
        X = self.constructX(imgs_set, lmks_in, sift)
        print("sift done!")
        X_r, A = self.processPCA(X)
        print("pca done!")
        X_r_i = np.insert(X_r, X_r.shape[0], np.zeros(X_r.shape[1])+1, axis=0)
        lmks_in = reshapeLmkCoords(lmks_in)
        lmks_gt = reshapeLmkCoords(lmks_gt)
        delta_s = lmks_gt - lmks_in
        R = self.computeR(X_r_i, delta_s)
        print("compute R done!")
        delta_s_0 = (R.dot(X_r_i)).T
        new_lmks = lmks_in + delta_s_0
        return new_lmks, R, A

    def fit(self, train_imgs_set, train_lmks_in, train_lmks_gt, nIter=5):
        print("------------ start training ------------")
        self.Es_tr = []
        self.Rs = []
        self.As = []
        new_lmks = train_lmks_in
        for i in range(nIter):
            print("--- " + str(i+1) + " epoche start ---")
            new_lmks, R, A = self.oneIterAlignment(train_imgs_set, train_lmks_gt, new_lmks)
            new_lmks = recoverLmkCoords(new_lmks)
            error = calculateError(train_lmks_gt, new_lmks)/len(train_lmks_gt)
            self.Es_tr.append(error)
            self.Rs.append(R)
            self.As.append(A)
            print("--- " + str(i+1) + " epoche end ---")
            showData_2(train_imgs_set[0], train_lmks_in[0], new_lmks[0])
        print("------------ end training ------------")
        
    def oneIterPredict(self, imgs_set, lmks_in, R, A):
        sift = cv2.SIFT_create()
        X = self.constructX(imgs_set, lmks_in, sift)
        print("sift done!")
        X_r = A.dot(X)
        X_r_i = np.insert(X_r, X_r.shape[0], np.zeros(X_r.shape[1])+1, axis=0)
        delta_s_p = (R.dot(X_r_i)).T
        print("predict done!")
        lmks_in = reshapeLmkCoords(lmks_in)
        lmks_predict = lmks_in + delta_s_p
        return lmks_predict

    def predict(self, test_imgs_set, lmks_in):
        print("------------ start predicting ------------")
        self.predictLmks = []
        lmks_predict = lmks_in
        for i in range(len(self.Rs)):
            print("--- " + str(i+1) + " epoche start ---")
            lmks_predict = self.oneIterPredict(test_imgs_set, lmks_predict, self.Rs[i], self.As[i])
            lmks_predict = recoverLmkCoords(lmks_predict)
            print("--- " + str(i+1) + " epoche end ---")
        print("------------ end predicting ------------")
        self.predictLmks = lmks_predict





            
    
