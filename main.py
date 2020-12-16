from typing import Any, Union, List

import cv2
import numpy as np
quantalama = 16
tsimilarity = 10 #euclid distance similarity threshhold
tdistance = 20 #euclid distance between pixels threshold
block_counter = 0
block_size = 8
image = cv2.imread('forged2_gj90.png')
acc_mask = cv2.imread('forged2_maske.png')
acc_mask_grey = cv2.cvtColor(acc_mask, cv2.COLOR_RGB2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
a = []
arr = np.array(gray)
maske = np.array(acc_mask_grey)
prediction_mask = np.zeros((arr.shape[0], arr.shape[1]))
column = arr.shape[1] - block_size
row = arr.shape[0] - block_size
dcts = np.empty((((column+1)*(row+1)), 18))

print("scanning & dct starting...")
for i in range(0, row): #satırlar geziyor
    for j in range(0, column): #sütunları geziyor

        blocks = arr[i:i+block_size,j:j+block_size]
        imf = np.float32(blocks) / 255.0  # float conversion/scale
        dst = cv2.dct(imf)  # the dct
        blocks = np.uint8(np.float32(dst) * 255.0 ) # convert back
        #zigzag tarama
        solution = [[] for k in range(block_size + block_size - 1)]
        for k in range(block_size):
            for l in range(block_size):
                sum = k + l
                if (sum % 2 == 0):
                    # add at beginning
                    solution[sum].insert(0, blocks[k][l])
                else:
                    # add at end of the list
                    solution[sum].append(blocks[k][l])


        for item in range(0,(block_size*2-1)):
            a += solution[item]

        a = np.asarray(a, dtype=np.float)
        a = np.array(a[:16])
        a = np.floor(a/quantalama)
        a = np.append(a, [i, j])

        np.copyto(dcts[block_counter], a)

        block_counter += 1
        a = []
        #solution zigzag çıktısı

print("scanning & dct over!")

#----------------------------------------------------------------------------------------

print("lexicographic ordering starting...")

dcts = dcts[~np.all(dcts == 0, axis=1)]
dcts = dcts[np.lexsort(np.rot90(dcts))]


print("lexicographic ordering over!")

#----------------------------------------------------------------------------------------

print("euclidean operations starting...")

sim_array = []
for i in range(0, block_counter):
    if i <= block_counter-10:
        for j in range(i+1, i+10):
            pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
            pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
            if pixelsim <= tsimilarity and pointdis >= tdistance:
                sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])
    else:
        for j in range(i+1, block_counter):
            pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
            pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
            if pixelsim <= tsimilarity and pointdis >= tdistance:
                sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])



print("euclidean operations over!")

#----------------------------------------------------------------------------------------

print("elimination starting...")

sim_array = np.array(sim_array)
delete_vec = []
vector_counter = 0
for i in range(0, sim_array.shape[0]):
    temp = [sim_array[i][4], sim_array[i][5]]
    check = all(item in delete_vec for item in [temp])
    if not check:
        for j in range(1, sim_array.shape[0]):
            if sim_array[i][4] == sim_array[j][4] and sim_array[i][5] == sim_array[j][5]:
                vector_counter += 1
        if vector_counter < 14:
            delete_vec.append(sim_array[i])
        vector_counter = 0

delete_vec = np.array(delete_vec)
delete_vec = delete_vec[~np.all(delete_vec == 0, axis=1)]
delete_vec = delete_vec[np.lexsort(np.rot90(delete_vec))]

for item in delete_vec:
    indexes = np.where(sim_array == item)
    unique, counts = np.unique(indexes[0], return_counts=True)
    for i in range(0, unique.shape[0]):
        if counts[i] == 6:
            sim_array = np.delete(sim_array,unique[i],axis=0)

print("elimination over!")

#----------------------------------------------------------------------------------------

print("painting starting...")

for i in range(0, sim_array.shape[0]):
    index1 = int(sim_array[i][0])
    index2 = int(sim_array[i][1])
    index3 = int(sim_array[i][2])
    index4 = int(sim_array[i][3])
    for j in range(0,7):
        for k in range(0,7):
            prediction_mask[index1+j][index2+k] = 255.0
            prediction_mask[index3+j][index4+k] = 255.0

print("painting over!")

#----------------------------------------------------------------------------------------

print("accuracy calculating...")

TP = 0.0
FP = 0.0
TN = 0.0
FN = 0.0

for i in range(0, prediction_mask.shape[0]):
    for j in range(0, prediction_mask.shape[1]):
        if prediction_mask[i][j] == maske[i][j]:
            if prediction_mask[i][j] == 0:
                TP += 1
            else:
                TN += 1
        else:
            if prediction_mask[i][j] == 0:
                FP += 1
            else:
                FN += 1

acc = ((TP+TN)/(TP+TN+FP+FN))*100
TPR = TP/(TP+FN)
FNR = 1 - TPR
TNR = TN/(TN+FP)
FPR = 1-TNR
print(acc, TPR, FNR, TNR, FPR)

print("accuracy calculated!")

#----------------------------------------------------------------------------------------

cv2.imshow("Prediction Mask", prediction_mask)
cv2.imshow("Real Mask", maske)
cv2.imshow('Original image', image)
cv2.imshow('Gray image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()



