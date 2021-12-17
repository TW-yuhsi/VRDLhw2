# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:31:57 2021

@author: ktpss

department: IMM
student number: 309653012
name: yuhsi, Chen
"""


import os
import cv2
import json


################################################################################
#                                path setup                                    #
################################################################################
# Use the results from your model to generate the output json file
testimgPath = '/home/yuhsi44165/NYCU/G2/VRDL/HW2/2021VRDL_HW2_datasets/images/test/'
data_listdir = os.listdir(testimgPath)
data_listdir.sort(key = lambda x: int(x[:-4]))
filepath = '/home/yuhsi44165/NYCU/G2/VRDL/HW2/yolov5/runs/detect/exp_yolo_road_det_2/labels/'

data = []
fail_detect_image = []

for i in data_listdir:
	i = i.replace('.png', '')
	print("----------------------------i: ", i, ".png----------------------------")
	#if not os.path.isfile(filepath+str(i)+'.txt'):
	#a = {"image_id" : "wrong","bbox": [(1,1,1,1)], "score": [0.5], "label": [0]}
	#assert os.path.isfile(filepath+str(i)+'.txt')
	if not os.path.isfile(filepath+str(i)+'.txt'):
		fail_detect_image.append(i)
		a = {"bbox": [(1,1,1,1)], "score": [0.5], "label": [0]}
		a = {"image_id" : int(i), "bbox":[1,1,1,1], 'score' : 0.5, 'category_id' : 0}
		data.append(a)
	else:
		f = open(filepath+str(i)+'.txt','r')
		print("open txt: ", filepath+str(i)+'.txt')

		contents = f.readlines()

		img_name = str(i)+'.png'
		im = cv2.imread(testimgPath + img_name)
		h, w, c = im.shape
		#print("image id: ", img_name, ", shape: (", h," , ",w,")")

		#a = {"image_id": str(i) ,"bbox": [], "score": 0, "label": 0}
		for content in contents:
			a = {"image_id" : int(i), "bbox":[]}
			content = content.replace('\n','')
			c = content.split(' ')
			print(c)

			w_center = w*float(c[1])
			h_center = h*float(c[2])
			width = w*float(c[3])
			height = h*float(c[4])
			left = float(w_center - width/2)
			right = float(w_center + width/2)
			top = float(h_center - height/2)
			bottom = float(h_center + height/2)

			#a['bbox'].append(tuple((top, left, bottom, right)))
			a['bbox'].append(left)
			a['bbox'].append(top)
			a['bbox'].append(width)
			a['bbox'].append(height)

			a['score'] = float(c[5])
			a['category_id'] = int(c[0])
			# print(a)
			data.append(a)


f.close()

#print(data)
print(fail_detect_image)

# Write the list to answer.json
ret = json.dumps(data)

with open('answer.json', 'w') as fp:
    fp.write(ret)	
