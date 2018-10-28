

import numpy as np
import cv2
img=cv2.imread("gai.JPG",0)
cv2.line(img,(100,100),(200,200),(255,0,0),4)
cv2.imwrite('gai1.JPG',img)
cv2.destroyAllWindows()﻿
