import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from kmeans import KMeans

# Importing dataset No. 1
df= pd.read_csv('Dataset1.csv')
X1 = df.iloc[:, [ 0, 1]].values
df = pd.read_csv('Dataset2.csv')
X2 = df.iloc[:, [0, 1]].values


# Clustering on the first dataet
fig = plt.figure(1)
a = []
i = range(2, 15)
for r in i:
    k = KMeans(K=r, max_iters=150)
    y_pred, cent = k.predict(X1)
    dist = k.distortion()
    a.append(dist)
plt.plot(i, a, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method ')
plt.show()

# CLusetering on the second dataset
a = []
i = range(2, 5)
for r in i:
    k = KMeans(K=r, max_iters=150)
    y_pred, cent = k.predict(X2)
    dist = k.distortion()
    a.append(dist)
    # k.plot()

fig = plt.figure(2)
plt.plot(i, a, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method ')
plt.show()

#################### image #####################
# Reading image
print("reading image ...")
img = cv.imread("imageSmall.png").astype(np.int32)

p_list = []
h, w, c = img.shape

for i in range(h):
    for j in range(w):
        p_list.append(img[i][j])
p_list = np.array(p_list)

# Kmeans with K = 2
print("kmean on image ...")
km = KMeans(K=2)
y_pred, cnt = km.predict(p_list)

print("making new image ...")
new_img = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        new_img[i][j] = y_pred[i*w + j]*255
new_img = new_img.astype(np.uint8)

print("saving new image ...")
cv.imwrite("compressed_2_levels.png", new_img)

# Kmeans with K = 16
print("kmean on image ...")
km = KMeans(K=16)
y_pred, cnt = km.predict(p_list)

print("making new image ...")
new_img = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        new_img[i][j] = y_pred[i*w + j]*255/16
new_img = new_img.astype(np.uint8)

print("saving new image ...")
cv.imwrite("compressed_16_levels.png", new_img)

# Mall Customers
df = pd.read_csv("Mall_Customers.csv")
df = df.replace({'Male': 1, 'Female': 0})

p_list = []
for index, row in df.iterrows():
    p_list.append(row.to_numpy()[1:])
p_list = np.array(p_list)

km = KMeans(K = 5) # Based on the elbow method results, the optimum K is 5
y_pred, cnt = km.predict(p_list)

