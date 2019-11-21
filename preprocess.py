import os
from glob import glob
from PIL import Image
import numpy as np
import cv2

'''
본 파일은 원본 이미지 데이터를 전처리하여 csv파일로 저장하는 파일입니다.
'''


'''
아래는 경로로 부터 각 파일의 이름 및 label명을 읽어 one-hot-encoding 후 반환
'''
# 경로로 부터 label을 추출하는 함수
def get_label_from_path(path):
    return str(path.split('\\')[-2])   ### emotion은 -1하면 될듯


# 이미지 파일 경로들로 부터 label들이 담긴 리스트를 구해서 반환
def get_label_list(path_list):
    label_list = []
    for p in path_list:
        label_list.append(get_label_from_path(p))
    return np.array(label_list).astype(int)

'''
아래 두 함수는 path로 부터 데이터 사진 파일명을 통해 grayscale로 로드 및 리사이즈 후 리스트로 반환하는 역할
'''


# load gray scale img and resize
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return resized_img


# 각 이미지 데이터 파일의 경로 리스트를 받아 이미지를 로드후 numpy 배열에 담아 반환
def get_img_data_list(path_list):
    result = []
    for path in path_list:
        img=read_image(path)
        result.append(img)
    return np.array(result)


# flatten 작업 ex) (n, 50, 50) -> (n, 2500)
def trans_3d_to_2d(arr):
    result=[]
    for i in arr:
        result.append(i.flatten())
    return np.array(result)


# Normalize function
def normalize(x):
    x=x.astype(int)
    min_val = 0
    max_val = 255
    x = (x - min_val) / (max_val - min_val)
    return x


# split train dataset & test dataset
def split_train_test(data):
    msk = np.random.rand(len(data)) < 0.7
    train = data[msk]
    test = data[~msk]
    return train, test


# save numpy array to csv
def save_array_to_csv(arr, fname):
    np.savetxt(fname, arr, delimiter=',')


def main():
    data_path_list = glob('data\\*\\*.png') # 모든 경로들을 list로 반환
    label_list = get_label_list(data_path_list) # label들이 담긴 list를 얻음
    data_list = get_img_data_list(data_path_list)   # image 데이터 들이 담긴 list를 얻음
    
    data_list = trans_3d_to_2d(data_list)   # flatten : (n, 50, 50) -> (n, 2500)
    data_list = normalize(data_list)    # 정규화
    
    
    merged_data_list = np.column_stack((data_list, label_list)) # merge data and label

    # split
    train, test = split_train_test(merged_data_list)
    print('train length: ' + str(len(train)))
    print('test length: ' + str(len(test)))

     #save to csv
    save_array_to_csv(train, 'train_data.csv')
    save_array_to_csv(test, 'test_data.csv')
    print('csv writing task finished')


if __name__=='__main__':
    main()
