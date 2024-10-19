import numpy as np

x1_list = np.random.rand(500, 1) # tạo mảng x gồm các số ngẫu nhiên

y_list = 4 + 10 * x1_list + 0.2 * np.random.randn(x1_list.shape[0], 1)
#với w_1 = 10 và w_2 = 4, tạo ra các giá trị y tương ứng nhưng bị lệch đi một tí

ones = np.ones((x1_list.shape[0], 1))
X = np.concatenate((ones, x1_list), axis = 1)
# tạo ma trân X

W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y_list))
# Ma thuật của đại số tuyến tính
print(W)