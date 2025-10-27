import cv2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


digits = load_digits()

# print(digits.data.shape)

"""
img = digits.data[1300]
img2 = np.reshape(img, (8, 8))
img2 = cv2.resize(img2, (128, 128))
cv2.imshow("Image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

kmeans = KMeans(n_clusters=10, random_state=0)

clusters = kmeans.fit_predict(digits.data)

# print(kmeans.cluster_centers_.shape)
centers = kmeans.cluster_centers_

centers = np.reshape(centers, (10, 8, 8))

for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(centers[i-1])

plt.show()