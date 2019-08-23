# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:20:50 2019

@author: lxyi
"""
import cv2
from sheet import get_answer_from_sheet

#首先是要获取到图片
cat = cv2.imread('./img/test.png')

#对图片的大小进行一个处理
cat = cv2.resize(cat,(600,500),interpolation=cv2.INTER_CUBIC)

#扫描出填涂在答题卡上的样子
get_answer_from_sheet(cat)


