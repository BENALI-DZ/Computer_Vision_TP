import cv2
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.datasets import mnist
 


img = cv2.imread('car01.jpg', cv2.IMREAD_GRAYSCALE)
print("Image shape:", img.shape)
plt.imshow(img, cmap='gray')
plt.title("Grayscale Image")
plt.show()
flattened = img.flatten()
print("Flattened shape:", flattened.shape)
normalized = img.astype('float32') / 255.0
plt.hist(normalized.ravel(), bins=50)
plt.title("Normalized Pixel Intensity Distribution")
plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.show()
