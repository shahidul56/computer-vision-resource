import cv2

img1 = cv2.imread("walk_1.jpg")
img2 = cv2.imread("walk_2.jpg")
subtraction = img1-img2
sub2 = cv2.subtract(img1, img2)
cv2.imshow("Subtraction", subtraction)
cv2.imshow("Sub 2", sub2)
cv2.waitKey(0)
cv2.destroyAllWindows()
