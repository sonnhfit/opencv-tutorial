import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('datatrain.png', 0)
imgNhanDang = cv2.imread('anhtessss44444.PNG', 0)
imgNhanDang = cv2.resize(imgNhanDang, (20, 20))
cv2.imwrite( 'anhtesst3.png', imgNhanDang);
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)] #cat tung anh nho tu anh to
#khong can cat nua


x = np.array(cells)
print (len(x))
x2 = np.array(imgNhanDang)

#tao du lieu train va du lieu test
train = x[:,:50].reshape(-1,400).astype(np.float32)
test2 = x2.reshape(-1,400).astype(np.float32)


#gan nhan cho du lieu train
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]

#nhan dang
knn = cv2.ml.KNearest_create()
knn.train(train, 0 ,train_labels)
kq1, kqChungTaCan, kq3, kq4 = knn.findNearest(test2, 5)


print( kqChungTaCan)










