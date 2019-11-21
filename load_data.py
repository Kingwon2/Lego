import numpy as np
import data_class

'''
본 파일은 csv 파일을 읽어 numpy array로 저장하여 반환하는 함수 관련 파일입니다.
'''

# one-hot-encode를 수행하는 함수
def one_hot_encode(x):
    encoded = np.zeros((len(x), 16))

    for idx, val in enumerate(x):
        encoded[int(idx)][int(val)] = 1

    return encoded

# data loading & preprocess(onehot, normalization ...)
def load_and_preprocess_data(file_path):
    origin_datas = np.loadtxt(fname=file_path, dtype=float, delimiter=',')

    # train data preprocessing
    x_datas = origin_datas[..., :-1]
    y_datas = origin_datas[..., -1:]
    y_datas = one_hot_encode(y_datas)   # one-hot-encoding

    merged_data = np.column_stack((x_datas,y_datas))

    # conversion np.array -> dataset class
    datas = data_class.Dataset(merged_data)

    return datas