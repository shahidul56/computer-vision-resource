

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
UBIT = "rgupta24"

# Importing the images
pic1 = cv2.imread("data/NotreDame1.jpg")
pic2 = cv2.imread("data/NotreDame2.jpg")


# Converting images to gray-scale
graypic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
graypic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

# Creating SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Finding keypoints and descriptors for Mountain images
keyp1, desc1 = sift.detectAndCompute(graypic1, None)
keyp2, desc2 = sift.detectAndCompute(graypic2, None)

# Drawing keypoints for Mountain images
keyImage1 = cv2.drawKeypoints(graypic1, keyp1, np.array([]), (255, 0, 0))
keyImage2 = cv2.drawKeypoints(graypic2, keyp2, np.array([]), (255, 0, 0))

plt.figure(figsize=(20, 20))
plt.imshow(keyImage1)
plt.show()
plt.figure(figsize=(20, 20))
plt.imshow(keyImage2)
plt.show()
cv2.imwrite('output/sift1.jpg', keyImage1)
cv2.imwrite('output/sift2.jpg', keyImage2)


# Brute-Force matching with SIFT descriptors
brutef = cv2.BFMatcher()

# Matching the keypoints with k-nearest neighbor (with k=2)
matches = brutef.knnMatch(desc1, desc2, k=2)

goodMatch = []
# Performing ratio test to find good matches
for m, n in matches:
    if m.distance < 0.75*n.distance:
        goodMatch.append(m)

# Drawing good matches
matchImage = cv2.drawMatches(
    pic1, keyp1, pic2, keyp2, goodMatch, np.array([]), (255, 0, 255), flags=2)

plt.figure(figsize=(20, 20))
plt.imshow(matchImage)
plt.show()
cv2.imwrite('output/matches_knn.jpg', matchImage)


# Getting source and destination points
srce_pts = np.float32(
    [keyp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
dest_pts = np.float32(
    [keyp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

# Finding Homography Matrix and mask
homographyMat, mask = cv2.findHomography(srce_pts, dest_pts, cv2.RANSAC, 5.0)
print(homographyMat)


# Converting the mask to a list
matchesMask = mask.ravel().tolist()

h, w = pic1.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

matchIndex = []
for i in range(len(matchesMask)):
    if (matchesMask[i]):
        matchIndex.append(i)

matchArray = []
for i in matchIndex:
    matchArray.append(goodMatch[i])

# Finding 10 random matches using inliers
np.random.seed(sum([ord(c) for c in UBIT]))
randomMatch = np.random.choice(matchArray, 10, replace=False)

# Defining draw parameters
draw_params = dict(matchColor=(0, 0, 255),
                   singlePointColor=None,
                   flags=2)

# Drawing the match image for 10 random points
matchImage = cv2.drawMatches(
    pic1, keyp1, pic2, keyp2, randomMatch, None, **draw_params)

plt.figure(figsize=(20, 20))
plt.imshow(matchImage)
plt.show()
cv2.imwrite('output/matches.jpg', matchImage)


h1, w1 = pic2.shape[:2]
h2, w2 = pic1.shape[:2]
pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
pts2_ = cv2.perspectiveTransform(pts2, homographyMat)
pts = np.concatenate((pts1, pts2_), axis=0)

# Finding the minimum and maximum coordinates
[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-xmin, -ymin]

# Translating
Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

# Warping the first image on the second image using Homography Matrix
result = cv2.warpPerspective(pic1, Ht.dot(
    homographyMat), (xmax-xmin, ymax-ymin))

#sresult[t[1]:h1+t[1], t[0]:w1+t[0]] = pic2

plt.figure(figsize=(20, 20))
plt.imshow(result)
plt.show()
cv2.imwrite('out/Homography_applied_image.jpg', result)
