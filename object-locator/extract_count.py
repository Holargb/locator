import scipy.io as sio

file_path = r'Dataset\mall_dataset\mall_gt.mat'

data = sio.loadmat(file_path)

count_matrix = data['count']

print(count_matrix)