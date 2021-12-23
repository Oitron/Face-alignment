# Face-alignment
Implementation of a cascade of regressions for facial landmarks localisation
****
### Dataset we used:
300W (300 Faces-In-The-Wild)   
**Link**: https://ibug.doc.ic.ac.uk/resources/300-W/

****
### Data pre-processing:
All images in trainset and testset are cropped with a bounding box of landmarks and resize to 128 x 128.  
Initialize the mean shape in each image and generate 10 random perturbations (in translation and scaling) around mean position for data augmentation. The amplitude of these perturbations will be ±20% and ±20px for scaling and translation respectively (only for trainset).  
All functions are present in **LoadData.py**

****
### Model construction:
Compute SIFT and decrease dimentions of features using PCA, as the prameter of the linear regression. 
The Supervised Descent Method(A cascade of regressions) is be impremented in **SDM.py** including **fit()** and **predict()** function.

****
### Result:
Five train iterations: **Blue**: fit, **Red**: initial(mean)  
![image](https://github.com/Oitron/Face-alignment/blob/main/output/train_iter_01.png)
![image](https://github.com/Oitron/Face-alignment/blob/main/output/train_iter_02.png)
![image](https://github.com/Oitron/Face-alignment/blob/main/output/train_iter_03.png)
![image](https://github.com/Oitron/Face-alignment/blob/main/output/train_iter_04.png)
![image](https://github.com/Oitron/Face-alignment/blob/main/output/train_iter_05.png)

Five results on testset: **Blue**: predict, **Red**: initial(mean)

****
### Source:
[1] Xuehan Xiong and Fernando De la Torre. “Supervised Descent Method and Its Applications to Face Alignment”. In: CVPR ’13. 2013, pp. 532–539.
