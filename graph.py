import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('./Output/baby.png')
image = image.astype('uint8')
# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the edges of the image
edges = cv2.Canny(image, 100, 200)

# Calculate the displacement per pixel
coordinates = np.where(edges > 0)[1]
# coordinates = coordinates.T
# print(coordinates[0])
# print(coordinates[1])
# print(y)
# print(edges[215][32])


plt.plot([i for i in range(1796)], coordinates)
plt.savefig("my_plot.png")

# Print the displacement per pixel
# cv2.imshow('Apple',coordinates)
# cv2.waitKey(0)
