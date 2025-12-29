import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('car01.jpg')
# Display using Matplotlib
plt.imshow(image)
plt.title("Original (BGR) image")
plt.show()
  
print(image.shape)
