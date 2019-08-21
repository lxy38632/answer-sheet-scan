# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:20:50 2019

@author: lxyi
"""
import cv2
#from settings import ANS_IMG_THRESHOLD, CNT_PERIMETER_THRESHOLD, CHOICE_IMG_THRESHOLD,ANS_IMG_DILATE_ITERATIONS,ANS_IMG_ERODE_ITERATIONS, CHOICE_IMG_DILATE_ITERATIONS,CHOICE_IMG_ERODE_ITERATIONS, CHOICE_MAX_AREA, CHOICE_CNT_COUNT, ANS_IMG_KERNEL,CHOICE_IMG_KERNEL, CHOICE_MIN_AREA
#from utils import  detect_cnt_again, get_init_process_img, get_max_area_cnt, get_ans,get_left_right, get_top_bottom, sort_by_row, sort_by_col, insert_null_2_rows, get_vertical_projective,get_h_projective
'''
from settings import CNT_PERIMETER_THRESHOLD,ANS_IMG_KERNEL,ANS_IMG_DILATE_ITERATIONS,\
ANS_IMG_ERODE_ITERATIONS,ANS_IMG_THRESHOLD,CHOICE_IMG_DILATE_ITERATIONS,CHOICE_IMG_ERODE_ITERATIONS,\
CHOICE_IMG_KERNEL,CHOICE_MIN_AREA,CHOICE_MAX_AREA
from utils import get_init_process_img, get_max_area_cnt,detect_cnt_again,\
get_vertical_projective,get_h_projective,get_left_right,get_top_bottom,\
sort_by_row,sort_by_col,insert_null_2_rows,get_ans
'''
from settings import *
from utils import *
from e import ContourPerimeterSizeError,PolyNodeCountError
from imutils import contours

#获取到图片
cat = cv2.imread('./test.png')
'''
cv2.imshow('temp1',cat)
cv2.waitKey(0)
展现出来的就是原来的大小
'''

cat = cv2.resize(cat,(600, 500),interpolation=cv2.INTER_CUBIC)
'''
cv2.imshow('temp2',cat)
cv2.waitKey(0)
对图片的大小进行设置
'''
#灰度化然后进行边缘检测，二值化等等一系列处理
base_img = cat
img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
img = get_init_process_img(img)
'''
cv2.imshow('temp3',img)
cv2.waitKey(0)
展示出来的就是黑白边框的样子
'''

#获取最大面积轮廓冰河图片大小作比较，看轮廓周长大小判断是否是答题卡的轮廓
cnt = get_max_area_cnt(img)
cnt_perimeter = cv2.arcLength(cnt,True)
base_img_perimeter = (base_img.shape[0] + base_img.shape[1]) * 2
if not cnt_perimeter > CNT_PERIMETER_THRESHOLD * base_img_perimeter:
    raise ContourPerimeterSizeError
'''
cv2.drawContours(base_img,[cnt],0,(0,255,0),1)
cv2.imshow('temp4',base_img)
cv2.waitKey(0)
展示的是将答题卡边框用绿线标出来的样子
'''

#计算多边形的顶点，并看是否是四个顶点
poly_node_list = cv2.approxPolyDP(cnt,cv2.arcLength(cnt,True) * 0.1, True)
if not poly_node_list.shape[0] == 4:
    raise PolyNodeCountError
    
#根据计算的多边形顶点继续处理图片，主要是是纠偏
processed_img = detect_cnt_again(poly_node_list, base_img)
'''
cv2.imshow('temp5',processed_img)
cv2.waitKey(0)
展示的是裁剪出只包含答题卡的部分
'''

#调整图片的亮度
processed_img = cv2.cvtColor(processed_img,cv2.COLOR_BGR2GRAY)
processed_img = cv2.GaussianBlur(processed_img,(9,9),0)
'''
cv2.imshow('temp6',processed_img)
cv2.waitKey(0)
'''

#通过二值化和膨胀服饰获得填涂区域
# ret, ans_img = cv2.threshold(processed_img, ANS_IMG_THRESHOLD[0], ANS_IMG_THRESHOLD[1], cv2.THRESH_BINARY_INV)
ans_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 35)
ans_img = cv2.dilate(ans_img, ANS_IMG_KERNEL, iterations=ANS_IMG_DILATE_ITERATIONS)
ans_img = cv2.erode(ans_img, ANS_IMG_KERNEL, iterations=ANS_IMG_ERODE_ITERATIONS)
ret, ans_img = cv2.threshold(ans_img, ANS_IMG_THRESHOLD[0], ANS_IMG_THRESHOLD[1], cv2.THRESH_BINARY_INV)
'''
cv2.imshow('temp7', ans_img)
cv2.waitKey(0)
展示的是，把填涂的样子展示出来
'''
#通过二值化和膨胀腐蚀获得选项框区域
choice_img = cv2.adaptiveThreshold(processed_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,10)
# ret, choice_img = cv2.threshold(processed_img, CHOICE_IMG_THRESHOLD[0], CHOICE_IMG_THRESHOLD[1],
#                                 cv2.THRESH_BINARY_INV)
'''
cv2.imshow('temp8', choice_img)
cv2.waitKey(0)
'''
#get_vertical_projective(choice_img)
#get_h_projective(choice_img)

for i in range(1):
    choice_img = cv2.dilate(choice_img,CHOICE_IMG_KERNEL, iterations=CHOICE_IMG_DILATE_ITERATIONS)
    choice_img = cv2.erode(choice_img, CHOICE_IMG_KERNEL, iterations=CHOICE_IMG_ERODE_ITERATIONS)
    
# get_vertical_projective(choice_img)
# get_h_projective(choice_img)
#cv2.imshow('temp9', choice_img)
#cv2.waitKey(0)


#查找选项框以及前面题号的轮廓
_,cnts, h = cv2.findContours(choice_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
question_cnts = []

temp1_ans_img = ans_img.copy()
for i,c in enumerate(cnts):
    #如果面积小于某值，则认为这个轮廓是选项框或是题号
    if CHOICE_MIN_AREA < cv2.contourArea(c) < CHOICE_MAX_AREA:
        cv2.drawContours(temp1_ans_img,cnts,i,(0,0,0),1)
        question_cnts.append(c)
#cv2.imshow('temp10',temp1_ans_img)
#cv2.waitKey(0)

question_cnts = get_left_right(question_cnts)
question_cnts = get_top_bottom(question_cnts)
   
temp2_ans_img = ans_img.copy()
for i in range(len(question_cnts)):
    cv2.drawContours(temp2_ans_img,question_cnts,i,(0,0,0),1)

#cv2.imshow('temp11',temp2_ans_img)
#cv2.waitKey(0)

# 如果轮廓小于特定值，重新扫描
# TODO 运用统计分析排除垃圾轮廓
# if len(question_cnts) != CHOICE_CNT_COUNT:
#     raise ContourCountError

# cv2.imshow('temp12', temp_ans_img)
# cv2.waitKey(0)

# 对轮廓之上而下的排序   
question_cnts, cnts_pos = contours.sort_contours(question_cnts, method="top-to-bottom")
rows = sort_by_row(list(cnts_pos))
cols = sort_by_col(list(cnts_pos))

#cv2.imshow('temp13', temp2_ans_img)
#cv2.waitKey(0)

insert_null_2_rows(cols, rows)
#获得答案

rows,res = get_ans(ans_img,rows)
if not res[0]:
    print res[1]
    cv2.imshow('temp1', temp2_ans_img)
    cv2.waitKey(0)
    print 'end'
else:
    print res

#cv2.imshow('temp1', temp2_ans_img)
#cv2.waitKey(0)
