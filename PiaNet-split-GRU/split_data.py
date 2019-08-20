#############reduce the number of training&valid data&test data##################
import os
import math
import numpy as np
import csv
import re
def atoi(s):
    return int(s) if s.isdigit() else s
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# savedct_path = "/media/lihuiyu/LITS/Preprocessed3/ct"
# savedseg_path = "/media/lihuiyu/LITS/Preprocessed3/seg/"
# savedct_path = "/home/lihuiyu/Data/LiTS/Preprocessed256_16/ct"
# savedseg_path = "/home/lihuiyu/Data/LiTS/Preprocessed256_16/seg"
savedct_path = "/home/lihuiyu/Data/LiTS/Preprocessed00/ct"
savedseg_path = "/home/lihuiyu/Data/LiTS/Preprocessed00/seg"

train_csv = './train.csv'
valid_csv = './valid.csv'
# test_csv = '/home/lihuiyu/Documents/Segmentation/LiTS-Vnet3D/dataprocess/test.csv'

#clear the exists file
if os.path.isfile(train_csv):
    os.remove(train_csv)
if os.path.isfile(valid_csv):
    os.remove(valid_csv)

ct_lists = os.listdir(savedct_path)
ct_lists.sort(key=natural_keys)
num_file = len(ct_lists)
ratio = 0.8
tn = math.ceil(num_file * ratio)

train_lists = ct_lists[0:tn]#attention:[0:num_train)
valid_lists = ct_lists[tn:num_file]

with open(train_csv, 'w') as file:
    w = csv.writer(file)
    # w.writerow(('Image','Label'))#attention: the first row defult to tile
    # or pd.read_csv(image_csv,header=None)#enable the first row by using defualt tile
    for name in train_lists:
        ct_name = os.path.join(savedct_path, name)
        seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
        w.writerow((ct_name,seg_name))

with open(valid_csv, 'w') as file:
    w = csv.writer(file)
    # w.writerow(('Image','Label'))#attention: the first row defult to tile
    # or pd.read_csv(image_csv,header=None)#enable the first row by using defualt tile
    for name in valid_lists:
        ct_name = os.path.join(savedct_path, name)
        seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
        w.writerow((ct_name,seg_name))
print(num_file)