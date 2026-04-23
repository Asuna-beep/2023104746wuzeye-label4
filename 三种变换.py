import cv2
import numpy as np

img = cv2.imread("5-1.jpg")
h, w = img.shape[:2]

# 相似变换
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 30, 0.8)
similar_img = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("similarity.jpg", similar_img)

# 仿射变换
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(img, M_affine, (w, h))
cv2.imwrite("affine.jpg", affine_img)

# 透视变换
pts1_p = np.float32([[50,50],[w-50,50],[50,h-50],[w-50,h-50]])
pts2_p = np.float32([[80,100],[w-10,50],[100,h-80],[w-50,h-20]])
M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
persp_img = cv2.warpPerspective(img, M_persp, (w, h))
cv2.imwrite("perspective.jpg", persp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()