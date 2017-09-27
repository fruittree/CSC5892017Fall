'''
Problem Set 1 Solution
Most of this solution is based on Yan Shi's submission
Please note that there are multiple correct solutions to each question, and the following demo script is just one of them.

'''
import numpy as np
from numpy import matlib
from scipy import misc
import cv2


import os
import matplotlib.pyplot as plt
import scipy
import skimage as io
from skimage import io


# === 2. Basic Matrix/Vector Manipulation ===
# === 2a. Define Matrix M and Vectors a, b, c in Python. ===

M = np.array(
            [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 2, 2]])
a = np.array([1, 1, 0])
b = np.array([-1, 2, 5])
c = np.array([0, 2, 3, 2])


print("%s") % "Matrix M:"
print(M)
print("%s") % "Vector a:"
print(a)
print("%s") % "Vector b:"
print(b)
print("%s") % "Vector c:"
print(c)

print("%s") % "--------------"



# === 2b. Find the dot product of vectors a and b ===

print("%s") % "The dot product is:"
aDotb = np.dot(a, b)
print(aDotb)

print ("%s") % "--------------"



# =============================================
a.shape = (3,1)
b.shape = (3,1)



# === 2c. Find the element wise product of a and b. ===

print("%s") % "Element-wise product:"
print((np.multiply(a, b)).T)

print("%s") % "--------------"



# === 2d. Find (a^Tb)Ma. ===
answer2d = np.dot(a.T, b) * np.dot(M, a)
print("%s") % "(a^Tb)Ma ="
print(answer2d)

print("%s") % "--------------"



# === 2e. Without using a loop, multiply each row of M element-wise by a. ===

newA = np.matlib.repmat(a.T, 4, 1)
answer2e = np.multiply(newA, M)
print(answer2e)

print("%s") % "--------------"



# === 2f. Without using a loop, sort all of the values of the new M from (e) in increasing order ===
sortedMatrix = answer2e.flatten()
sortedMatrix.sort()
sortedMatrix.shape = (answer2e.shape[0], answer2e.shape[1])

print(sortedMatrix)

print("%s %s %s") % ("--------------", " END PART 2 ", "--------------")
print("%s %s %s") % ("--------------", " BEGIN PART 3 ", "--------------")



# === 3a + b. Read in the images, image1.jpg and image2.jpg. Convert double precision and normamlized. ===
# Read in, double precision, aka float64

image1 = np.float64(misc.imread('image1.jpg', flatten = 1, mode='F'))
image2 = np.float64(misc.imread('image2.jpg', flatten = 1, mode='F'))


#normalization
normalizedImage1 = np.zeros((720, 652))
normalizedImage1 = cv2.normalize(image1, normalizedImage1, 0, 1, cv2.NORM_MINMAX)
normalizedImage2 = np.zeros((720, 652))
normalizedImage2 = cv2.normalize(image2, normalizedImage2, 0, 1, cv2.NORM_MINMAX)



# === 2c. Add the images together and renormalize them to have minimum/max of 0, 1. ===
smashed = np.zeros((720, 652))
smashed = cv2.normalize((normalizedImage1 + normalizedImage2), smashed, 0, 1, cv2.NORM_MINMAX)

# saving
# io.imsave('added_images.jpg', smashed)

# displaying
# plt.imshow(smashed, cmap=plt.cm.gray)
# plt.show()



# === 2d. Create a new image such that the left half of the image is the left
# half of image1 and the right half of the image is the right half of image 2. ===
croppedImage1 = normalizedImage1[0:720, 0:652/2]
croppedImage2 = normalizedImage2[:, 652/2:]

# display crops
# plt.imshow(croppedImage1)
# plt.show()
# plt.imshow(croppedImage2)
# plt.show()

newImage = np.concatenate((croppedImage1, croppedImage2), axis=1)
# saving
# io.imsave('concatenated.jpg', newImage)


# display newImage
# plt.imshow(newImage, cmap=plt.cm.gray)
# plt.show()



# === 2e. Using a for loop, create a new image such that every odd numbered row is
# the corresponding row from image1 and that every even row is the corresponding row from image2. ===
frankensteinImage = np.empty((720, 652))
for row in range(len(normalizedImage1)):
    if row % 2 == 1: #odd
        frankensteinImage[row] = normalizedImage1[row]

    else: #even
        frankensteinImage[row] = normalizedImage2[row]

plt.imshow(frankensteinImage)
plt.show()



# === 2f. Accomplish the same task as part e without using a for-loop (functions
#reshape and repmat may be helpful here).  ===
oddMatrix = normalizedImage1[1::2]
evenMatrix = normalizedImage2[::2]
frankensteinImage2 = np.empty((720, 652))
frankensteinImage2[1::2] = oddMatrix
frankensteinImage2[::2] = evenMatrix

# saving image
# frankensteinImage2 = misc.toimage(frankensteinImage2)
# frankensteinImage2.save('frankensteinImage2.jpg')


# === 2g. Convert the result from part f to a grayscale image. Display the grayscale
# image with a title in your report. ===
plt.imshow(frankensteinImage2, cmap=plt.cm.gray)
plt.title("Two Birds in One")
plt.axis('off')
plt.show()


# print("\n%s") % "Stats on Images"
# print(image1.shape)
# print(image2.shape)
# print(image1.dtype)
# print(image2.dtype)
# print(image1.max())
# print(image1.min())
# print(normalizedImage1.max())
# print(normalizedImage1.min())
# print(normalizedImage2.max())
# print(normalizedImage2.min())
# print(image2.max())
# print(image2.min())

print("%s %s %s") % ("--------------", " END PART 3 ", "--------------")
print("%s %s %s") % ("--------------", " BEGIN PART 4 ", "--------------")

folder = os.listdir('George_W_Bush')

# to store average later
average_array = np.zeros((250, 250, 3), dtype=np.float64)

# images_array requires an array that holds the multiple arrays of the images
images_array = np.float64(np.array([np.array(io.imread('George_W_Bush/' + fname)) for fname in folder]))

# use numpy's provided mean function to calculate average (average_array =  average_array + images_array / total
# would have worked as well ; don't forget to cast back into uint8 in order to properly display images)
# average_array = np.array(np.mean(images_array, axis=0))
average_array_type = np.array(np.mean(images_array, axis=0), dtype=np.uint8)
# plt.imshow(average_array, cmap=plt.cm.gray)
# plt.show()
# plt.imshow(average_array_type, cmap=plt.cm.gray)
# plt.show()

# saving image
# io.imsave('average_image_result.jpg', average_array_type)

# to display all the images on plots
plt.figure(figsize=(20, 4))
plt.subplot(141)

# Added together
plt.imshow(smashed, cmap=plt.cm.gray)
plt.axis('off')
plt.title('added together', fontsize=20)
plt.subplot(142)

# 2 crops together
plt.imshow(newImage, cmap=plt.cm.gray)
plt.axis('off')
plt.title('2 crops together', fontsize=20)
plt.subplot(143)

# For every row
plt.imshow(frankensteinImage2, cmap=plt.cm.gray)
plt.axis('off')
plt.title('2 birds in one', fontsize=20)
plt.subplot(144)

# Average Face
plt.imshow(average_array_type)
plt.axis('off')
plt.title('the average face', fontsize=20)
plt.subplot(144)

plt.show()
