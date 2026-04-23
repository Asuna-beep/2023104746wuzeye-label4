import cv2 # pyright: ignore[reportMissingImports]
import numpy as np

img = cv2.imread("5-1-2.jpg")

pts1 = np.float32([(435,85),(1270,367),(30,438),(1016,995)])
width, height = 420, 594
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

M = cv2.getPerspectiveTransform(pts1, pts2)
corrected = cv2.warpPerspective(img, M, (width, height))
corrected = cv2.rotate(corrected, cv2.ROTATE_180)

cv2.imwrite("a4_corrected.jpg", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()