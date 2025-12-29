import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('car01.jpg')
# Display using Matplotlib
plt.imshow(image)
plt.title("Original (BGR) image")
plt.show()
  
print(image.shape)


![car01](https://github.com/user-attachments/assets/5c21ad51-ba6d-424a-a252-3393ddc7211a)

