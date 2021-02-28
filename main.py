# Singular Value Decomposition
# Given image was alreaady in greyscale
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = np.array(Image.open(r"image.jpg"))
# normalize the intensity values in each pixel
image = image/255

row, col, _ = image.shape
print("Pixels: ", row, " * ", col)

# figure1 = plt.figure(figsize=(10, 5))
# x = figure1.add_subplot(1, 1, 1)
# imageplot = plt.imshow(image)
# x.set_title('Cat')
# plt.show()
# image remains same as it is already in greyscale
# next step is to sepearte greyscale values for each color. So, we will be working with 3 matrices with 2 dimension, and perform SVD decomposition for each color seperately.
# using k = 100

imageRED = image[:, :, 0]
imageGREEN = image[:, :, 1]
imageBLUE = image[:, :, 2]

# originalBYTES = image.nbytes
# print("To store the original image we need space in bytes is: ", originalBYTES)

# performing SVD decomposition, Omega is written as D
U_Red, D_Red, V_Red = np.linalg.svd(imageRED, full_matrices=True)
U_Green, D_Green, V_Green = np.linalg.svd(imageGREEN, full_matrices=True)
U_Blue, D_Blue, V_Blue = np.linalg.svd(imageBLUE, full_matrices=True)

# bytes_to_be_stored = sum([matrix.nbytes for matrix in [U_Red, D_Red, V_Red, U_Green, D_Green, V_Green, U_Blue, D_Blue, V_Blue]])
# print("The matrices have total size in bytes : ", bytes_to_be_stored)

k = 100

U_Red_K = U_Red[:, 0:k]
D_Red_K = D_Red[0:k]
V_Red_K = V_Red[0:k, :]
U_Green_K = U_Green[:, 0:k]
D_Green_K = D_Green[0:k]
V_Green_K = V_Green[0:k, :]
U_Blue_K = U_Blue[:, 0:k]
D_Blue_K = D_Blue[0:k]
V_Blue_k = V_Blue[0:k, :]

# compressedBYTES = sum([matrix.nbytes for matrix in [U_Red_K, D_Red_K, V_Red_K, U_Green_K, D_Green_K, V_Green_K, U_Blue_K, D_Blue_K, V_Blue_k]])
# print("Compressed bytes with k value 100 is : ", compressedBYTES)

# ratio = compressedBYTES/originalBYTES
# print("Ratio of the bytes : ", ratio)

# approximate matrices for each color and merging them. if the intensity of pixel is out of [0,1] this range, we need to fix it

image_red_approx = np.dot(U_Red_K, np.dot(np.diag(D_Red_K), V_Red_K))
image_green_approx = np.dot(U_Green_K, np.dot(np.diag(D_Green_K), V_Green_K))
image_blue_approx = np.dot(U_Blue_K, np.dot(np.diag(D_Blue_K), V_Blue_k))

image_reconstruct = np.zeros((row, col, 3))

image_reconstruct[:, :, 0] = image_red_approx
image_reconstruct[:, :, 1] = image_green_approx
image_reconstruct[:, :, 2] = image_blue_approx

image_reconstruct[image_reconstruct < 0] = 0
image_reconstruct[image_reconstruct > 1] = 1

figure1 = plt.figure(figsize=(10, 5))
y = figure1.add_subplot(1, 1, 1)
imageplot = plt.imshow(image_reconstruct)

y.set_title('Cat with best rank {} approximation'.format(k))
plt.show()

