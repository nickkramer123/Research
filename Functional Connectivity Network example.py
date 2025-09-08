import numpy as np
from sklearn.metrics import mutual_info_score

N0, N1 = data.shape #Elements are rows, frames are columns.
mi_mat = np.zeros((N0, N0))
for i in range(N0):
    for j in range(i): #Only need the triangle, since MI is symmetric.
        mi = mutual_information(data[i,:], data[j,:])
        mi_mat[i,j] = mi
        mi_mat[j,i] = m


