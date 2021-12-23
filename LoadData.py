import os
import shutil  

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean

from Tools import repeatLmk, reshapeLmkCoords, recoverLmkCoords


class loadData:
    def _init_(self):
        self.train_imgs = []
        self.train_lmks = []

        self.mean = []
        self.perurbs = []

        self.test_imgs = []
        self.test_lmks = []


    #get trainset/testset imgs/landmarks paths
    def loadDataPath(self, filepath):
        datapath = []
        fp = open(filepath)
        for line in fp.readlines():
            datapath.append(line.strip("\n"))
        return np.array(datapath)

    #read a file of landmarks，convet to coords
    def readLmk(self, filepath):
        landmarks = []
        fp = open(filepath)
        for line in fp.readlines():
            coord = line.strip("\n").split(" ")
            x = coord[0]
            y = coord[1]
            landmarks.append([float(x), float(y)])
        return landmarks

    #load train/test data
    def loadDataSet(self, dataSetPath, imgListFileName, lmkListFileName, isTrainset):
        imgs = []
        lmks = []
        imgPaths = self.loadDataPath(dataSetPath + '/' + imgListFileName)
        lmkPaths = self.loadDataPath(dataSetPath + '/' + lmkListFileName)
        if len(imgPaths) != len(lmkPaths):
            print("Data Loading failed: images and labels can't match! Please check these two files")
            print(dataSetPath + '/' + imgListFileName)
            print(dataSetPath + '/' + lmkListFileName)
            return 
        for i in range(len(imgPaths)):
            img = cv2.imread(dataSetPath + '/' + imgPaths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lmk = self.readLmk(dataSetPath + '/' + lmkPaths[i])
            imgs.append(img)
            lmks.append(lmk)
        if isTrainset:
            self.train_imgs = np.array(imgs, dtype=object)
            self.train_lmks = np.array(lmks)
        else:
            self.test_imgs = np.array(imgs, dtype=object)
            self.test_lmks = np.array(lmks)

    #Compute the parameters of the bounding box of the facial landmarks
    def computeBox(self, landmarks):
        landmarksT = landmarks.copy().T
        x_max = np.max(landmarksT[0])
        x_min = np.min(landmarksT[0])
        y_max = np.max(landmarksT[1])
        y_min = np.min(landmarksT[1])
        topLeft = [x_min, y_min]
        bottomRight = [x_max, y_max]
        return topLeft, bottomRight

    def expendBox(self, img, landmarks):
        topLeft, bottomRight = self.computeBox(landmarks)
        width = bottomRight[0] - topLeft[0] + 1
        height = bottomRight[1] - topLeft[1] + 1
        exp_w = width*0.3
        exp_h = height*0.3
        topLeft[0] -= exp_w
        topLeft[1] -= exp_h
        bottomRight[0] += exp_w + 1
        bottomRight[1] += exp_h + 1
        #bounding check
        if topLeft[0] < 0:
            topLeft[0] = 0
        if topLeft[1] < 0:
            topLeft[1] = 0
        if bottomRight[0] > img.shape[1]:
            bottomRight[0] = img.shape[1]
        if bottomRight[1] > img.shape[0]:
            bottomRight[1] = img.shape[0]
        return list(map(int, topLeft)), list(map(int, bottomRight))

    def cropAndResize(self, img, landmarks, width=128, height=128):
        topLeft, bottomRight = self.expendBox(img, landmarks)
        box_width = bottomRight[0] - topLeft[0] + 1
        box_height = bottomRight[1] - topLeft[1] + 1
        #recalculate landmarks
        landmarksT = landmarks.copy().T
        landmarksT[0] -= topLeft[0]
        landmarksT[0] *= width/box_width
        landmarksT[1] -= topLeft[1]
        landmarksT[1] *= height/box_height
        landmarks_rs = landmarksT.T
        #crop and resize img
        img_cr = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        img_rs = cv2.resize(img_cr, (width, height))
        return img_rs, landmarks_rs

    def optimizeData(self, width=128, height=128):
        ta_imgs = self.train_imgs
        ta_lmks = self.train_lmks
        te_imgs = self.test_imgs
        te_lmks = self.test_lmks
        self.train_imgs = list(self.train_imgs)
        self.train_lmks = list(self.train_lmks)
        self.test_imgs = list(self.test_imgs)
        self.test_lmks = list(self.test_lmks)
        self.train_imgs.clear()
        self.train_lmks.clear()
        self.test_imgs.clear()
        self.test_lmks.clear()
        for i in range(len(ta_imgs)):
            img_rs, lmk_rs = self.cropAndResize(ta_imgs[i], ta_lmks[i], width, height)
            self.train_imgs.append(img_rs)
            self.train_lmks.append(lmk_rs)
        for i in range(len(te_imgs)):
            img_rs, lmk_rs = self.cropAndResize(te_imgs[i], te_lmks[i], width, height)
            self.test_imgs.append(img_rs)
            self.test_lmks.append(lmk_rs)

    def computeMeanLmk(self):
        self.mean = np.mean(np.array(self.train_lmks), axis=0)
    
    def generatePerurbs(self, nbPerurb=10, bias_t=20, bias_s=0.2):
        perurbs = []
        translation = np.round(np.random.randn(nbPerurb,2)*2*bias_t-bias_t) # random translation from normal destribution +-20pixels
        scale = np.random.randn(nbPerurb,2)*2*bias_s-bias_s+1 # random factor from normal destribution +- 0.2
        for i in range(len(translation)):
            perurb = scale[i] * (self.mean + translation[i])
            perurbs.append(perurb)
        self.perurbs = np.array(perurbs)

    def saveData(self, savepath, filename, isTrainset):
        imgs_set = []
        lmks_set = []
        #clear all the files in savepath
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        #create directory
        path = savepath
        if isTrainset:
            path += "/trainset/"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
            imgs_set = self.train_imgs
            lmks_set = self.train_lmks
        else:
            path += "/testset/"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
            imgs_set = self.test_imgs
            lmks_set = self.test_lmks
        #save
        if len(imgs_set) != len(lmks_set):
            print("Data can't be saved, because image and landmarks can't be match!")
        imgpath = path + "images/"
        lmkpath = path + "landmarks/"
        os.mkdir(imgpath)
        os.mkdir(lmkpath)
        for i in range(len(imgs_set)):
            img = imgs_set[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(imgpath + filename + "_" + str(i) + ".jpg", img)
            np.save(lmkpath + filename + "_" + str(i) + ".npy", lmks_set[i])
        return imgpath, lmkpath

    def saveMeanAndPerurbs(self, savepath, filename):
        perurbpath = savepath + "/perurb/"
        meanpath = savepath + "/mean/"
        if os.path.exists(perurbpath):
            #clear all the files in savepath
            shutil.rmtree(perurbpath)
        if os.path.exists(meanpath):
            shutil.rmtree(meanpath)
        #create directory
        os.mkdir(perurbpath)
        os.mkdir(meanpath)
        np.save(meanpath + filename + "_m" + ".npy", self.mean)
        for i in range(len(self.perurbs)):
            np.save(perurbpath + filename + "_p_" + str(i) + ".npy", self.perurbs[i])
        return meanpath, perurbpath

    def loadPredData(self, imgpath, lmkpath, setToGray, isTrainset):
        imgs = []
        lmks = []
        imgfiles = os.listdir(imgpath)
        lmkfiles = os.listdir(lmkpath)
        if len(imgfiles) != len(lmkfiles):
            print("Data load failed: ")
            print(imgpath)
            print(lmkpath)
        for i in range(len(imgfiles)):
            img = cv2.imread(imgpath + imgfiles[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if setToGray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)
            lmks.append(np.load(lmkpath + lmkfiles[i]))
        if isTrainset:
            self.train_imgs = np.array(imgs)
            self.train_lmks = np.array(lmks)
        else:
            self.test_imgs = np.array(imgs)
            self.test_lmks = np.array(lmks)

    def loadMeanAndPerurbs(self, meanpath, perurbpath):
        mean = []
        perurbs = []
        meanfiles = os.listdir(meanpath)
        mean = np.load(meanpath + meanfiles[0])
        perurbfiles = os.listdir(perurbpath)
        for perurb in perurbfiles:
            perurbs.append(np.load(perurbpath + perurb))
        self.mean = np.array(mean)
        self.perurbs = np.array(perurbs)


    def augmentDataset(self, isTrainset):
        if isTrainset:
            imgs_set = self.train_imgs
            lmks_set = self.train_lmks
            lmks_mean = repeatLmk(self.mean, len(self.train_lmks))
            for perurb in self.perurbs:
                perurb_s = repeatLmk(perurb, len(self.train_lmks))
                lmks_mean = np.concatenate((lmks_mean, perurb_s), axis=0)
                lmks_set = np.concatenate((lmks_set, self.train_lmks), axis=0)
                imgs_set = np.concatenate((imgs_set, self.train_imgs), axis=0)
            return imgs_set, lmks_set, lmks_mean
        else:
            lmks_mean = repeatLmk(self.mean, len(self.test_lmks))
            imgs_set = self.test_imgs
            lmks_set = self.test_lmks
            return imgs_set, lmks_set, lmks_mean



        


    


