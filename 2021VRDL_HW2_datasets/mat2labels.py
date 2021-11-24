# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 02:19:42 2021

@author: ktpss

department: IMM
student number: 309653012
name: yuhsi, Chen
"""




import zipfile
import os
import shutil
import cv2
import h5py
import random


random.seed(309653012)




################################################################################
#                                   folders                                    #
################################################################################
path = '/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/'

if os.path.exists(path + 'images') and os.path.isdir(path + 'images'):
    shutil.rmtree(path + 'images')
if os.path.exists(path + 'labels') and os.path.isdir(path + 'labels'):
    shutil.rmtree(path + 'labels')

with zipfile.ZipFile('/home/yuhsi44165/NYCU/G2/VRDL/HW2/train.zip', 'r') as zip_ref:
    zip_ref.extractall(path + 'images')

with zipfile.ZipFile('/home/yuhsi44165/NYCU/G2/VRDL/HW2/test.zip', 'r') as zip_ref:
    zip_ref.extractall(path + 'images')

try:
    os.mkdir(path + 'images/val')
    os.mkdir(path + 'labels')
    os.mkdir(path + 'labels/train')
    os.mkdir(path + 'labels/val')
except OSError as error: 
    print(error)




################################################################################
#                                  functions                                   #
################################################################################
def get_name(index, hdf5_data):
    name_ref = hdf5_data['/digitStruct/name'][index].item()
    return ''.join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item_ref = hdf5_data['/digitStruct/bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int)
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs




if __name__ == "__main__":
    print('start converting...')

    indexList = [i for i in range(33402)]
    randomList = random.sample(indexList, int(33402/5))
    
    with h5py.File('/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/images/train/digitStruct.mat') as hdf5_data:
        for i in range(33402):
            img_name = get_name(i, hdf5_data)
            #print(img_name)
            im = cv2.imread('/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/images/train/'+img_name)
            h, w, c = im.shape
            arr = get_bbox(i, hdf5_data)

            if i in randomList:
                os.replace('/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/images/train/'+img_name, '/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/images/val/'+img_name)

                fp = open('/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/labels/val/'+img_name.replace('.png','.txt'), 'w')
                arr_l = len(arr['label'])
                for idx in range(arr_l):
                    label = arr['label'][idx]
                    if label==10:
                        label = 0
                    _l = arr['left'][idx]
                    _t = arr['top'][idx]
                    _w = arr['width'][idx]
                    if (_l+_w)>w:
                        _w = w-_l-1
                    _h = arr['height'][idx]
                    if (_t+_h)>h:
                        _h = h-_t-1
                    # print(w, h, _l, _t, _w , _h)
                    x_center = (_l + _w/2)/w
                    y_center = (_t + _h/2)/h
                    bbox_width = _w/w
                    bbox_height = _h/h
                    # print(label, x_center, y_center, bbox_width, bbox_height)
                    s = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(bbox_width)+' '+str(bbox_height)
                    if idx!=(arr_l-1):
                        s += '\n'
                    fp.write(s)
                fp.close()

            else:
                fp = open('/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/labels/train/'+img_name.replace('.png','.txt'), 'w')
                arr_l = len(arr['label'])
                for idx in range(arr_l):
                    label = arr['label'][idx]
                    if label==10:
                        label = 0
                    _l = arr['left'][idx]
                    _t = arr['top'][idx]
                    _w = arr['width'][idx]
                    if (_l+_w)>w:
                        _w = w-_l-1
                    _h = arr['height'][idx]
                    if (_t+_h)>h:
                        _h = h-_t-1
                    # print(w, h, _l, _t, _w , _h)
                    x_center = (_l + _w/2)/w
                    y_center = (_t + _h/2)/h
                    bbox_width = _w/w
                    bbox_height = _h/h
                    # print(label, x_center, y_center, bbox_width, bbox_height)
                    s = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(bbox_width)+' '+str(bbox_height)
                    if idx!=(arr_l-1):
                        s += '\n'
                    fp.write(s)
                fp.close()
    print('finished!')

