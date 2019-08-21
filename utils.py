# coding=utf-8
import math
from collections import Counter

import numpy as np
import cv2
from imutils import auto_canny, contours
from e import PolyNodeCountError
from score import score
from settings import CHOICES, SHEET_AREA_MIN_RATIO, PROCESS_BRIGHT_COLS, PROCESS_BRIGHT_ROWS, BRIGHT_VALUE,CHOICE_COL_COUNT, CHOICES_PER_QUE, WHITE_RATIO_PER_CHOICE, MAYBE_MULTI_CHOICE_THRESHOLD, CHOICE_CNT_COUNT, test_ans,ORIENT_CODE


def get_corner_node_list(poly_node_list):
    """
    获得多边形四个顶点的坐标
    :type poly_node_list: ndarray
    :return: tuple
    """
    center_y, center_x = (np.sum(poly_node_list, axis=0) / 4)[0]
    top_left = bottom_left = top_right = bottom_right = None
    for node in poly_node_list:
        x = node[0, 1]
        y = node[0, 0]
        if x < center_x and y < center_y:
            top_left = node
        elif x < center_x and y > center_y:
            bottom_left = node
        elif x > center_x and y < center_y:
            top_right = node
        elif x > center_x and y > center_y:
            bottom_right = node
    return top_left, bottom_left, top_right, bottom_right


def detect_cnt_again(poly, base_img):
    """
    继续检测已截取区域是否涵盖了答题卡区域
    :param poly: ndarray
    :param base_img: ndarray
    :return: ndarray
    """
    # 该多边形区域是否还包含答题卡区域的flag
    flag = False

    # 计算多边形四个顶点，并且截图，然后处理截取后的图片
    top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
    roi_img = get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right)
    img = get_init_process_img(roi_img)

    # 获得面积最大的轮廓
    cnt = get_max_area_cnt(img)

    # 如果轮廓面积足够大，重新计算多边形四个顶点
    if cv2.contourArea(cnt) > roi_img.shape[0] * roi_img.shape[1] * SHEET_AREA_MIN_RATIO:
        flag = True
        poly = cv2.approxPolyDP(cnt, cv2.arcLength((cnt,), True) * 0.1, True)
        top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
        if not poly.shape[0] == 4:
            raise PolyNodeCountError

    # 多边形顶点和图片顶点，主要用于纠偏
    base_poly_nodes = np.float32([top_left[0], bottom_left[0], top_right[0], bottom_right[0]])
    base_nodes = np.float32([[0, 0],
                            [base_img.shape[1], 0],
                            [0, base_img.shape[0]],
                            [base_img.shape[1], base_img.shape[0]]])
    transmtx = cv2.getPerspectiveTransform(base_poly_nodes, base_nodes)

    if flag:
        img_warp = cv2.warpPerspective(roi_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    else:
        img_warp = cv2.warpPerspective(base_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    return img_warp


def get_init_process_img(roi_img):
    """
    对图片进行初始化处理，包括，梯度化，高斯模糊，二值化，腐蚀，膨胀和边缘检测
    :param roi_img: ndarray
    :return: ndarray
    """
    h = cv2.Sobel(roi_img, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(roi_img, cv2.CV_32F, 1, 0, -1)
    img = cv2.add(h, v)
    img = cv2.convertScaleAbs(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    img = auto_canny(img)
    return img


def get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right):
    """
    截取合适的图片区域
    :param base_img: ndarray
    :param bottom_left: ndarray
    :param bottom_right: ndarray
    :param top_left: ndarray
    :param top_right: ndarray
    :return: ndarray
    """
    min_v = top_left[0, 1] if top_left[0, 1] < bottom_left[0, 1] else bottom_left[0, 1]
    max_v = top_right[0, 1] if top_right[0, 1] > bottom_right[0, 1] else bottom_right[0, 1]
    min_h = top_left[0, 0] if top_left[0, 0] < top_right[0, 0] else top_right[0, 0]
    max_h = bottom_left[0, 0] if bottom_left[0, 0] > bottom_right[0, 0] else bottom_right[0, 0]
    roi_img = base_img[min_v + 10:max_v - 10, min_h + 10:max_h - 10]
    return roi_img


def get_max_area_cnt(img):
    """
    获得图片里面最大面积的轮廓
    :param img: ndarray
    :return: ndarray
    此函数中的第一句，容易导致ValueError :too many values to umpack
    问题原因：返回值的个数发生改变
    解决的方法：前面加 ‘_’ 代表用不到的参数
    """
    _,cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    return cnt


def get_ans(ans_img, rows):
    # 选项个数加上题号
    interval = get_item_interval()   #5
    my_score = 0

    items_per_row = get_items_per_row()  #3
    ans = []
    for i, row in enumerate(rows):
        # 从左到右为当前题目的气泡轮廓排序，然后初始化被涂画的气泡变量
        for k in range(items_per_row):
            print '======================================='
            percent_list = []
            for j, c in enumerate(row[1 + k * interval:interval + k * interval]):
                try:
                    # 获得选项框的区域
                    new = ans_img[c[1]:(c[1] + c[3]), c[0]:(c[0] + c[2])]
                    # 计算白色像素个数和所占百分比
                    white_count = np.count_nonzero(new)
                    percent = white_count * 1.0 / new.size
                except IndexError:
                    percent = 1
                percent_list.append({'col': k + 1, 'row': i + 1, 'percent': percent, 'choice': CHOICES[j]})

            percent_list.sort(key=lambda x: x['percent'])
            choice_pos_n_ans = [percent_list[0]['row'], percent_list[0]['col']]
            choice_pos = (percent_list[0]['row'], percent_list[0]['col'])
            # if percent_list[1]['percent'] < 0.6 or (percent_list[1]['percent'] < WHITE_RATIO_PER_CHOICE and \
            #                 abs(percent_list[1]['percent'] - percent_list[0]['percent']) < MAYBE_MULTI_CHOICE_THRESHOLD):
            #     print u'第%s排第%s列的作答：可能多涂了选项' % choice_pos
            #     print u"第%s排第%s列的作答：%s" % choice_pos_n_ans
            #     ans.append(percent_list[0]['choice'])
            _ans = ''
            for percent in percent_list:
                if percent['percent'] < 0.8:
                    _ans += percent['choice']
            ans.append(''.join(sorted(list(_ans))))
            #print ans
            choice_pos_n_ans.append(''.join(sorted(list(_ans))))
            print u'第{0}排第{1}列的作答：{2}'.format(*choice_pos_n_ans)
            # elif percent_list[0]['percent'] < WHITE_RATIO_PER_CHOICE:
            #     # key = (percent_list[0]['row'] - 1) * 3 + percent_list[0]['col']
            #     # my_score += 1 if score.get(key) == percent_list[0]['choice'] else 0
            #     # print 1 if score.get(key) == percent_list[0]['choice'] else 0
            #     print u"第%s排第%s列的作答：%s" % choice_pos_n_ans
            #     print percent_list[0]['percent']
            #     ans.append(percent_list[0]['choice'])
            # else:
            #     print u"第%s排第%s列的作答：可能没有填涂" % choice_pos
            #     print percent_list[0]['percent']
            #     ans.append(None)
    
    print '=====总分========'
    return rows, test_is_eq(ans, test_ans)


def test_is_eq(ans, test_ans):
    count = 0
    for i, a in enumerate(ans):
        if a != test_ans[i]:
            print i / 4 + 1, i % 4, a
            count += 1
    if count:
        return False, count
    return True, count


def get_items_per_row():
    items_per_row = CHOICE_COL_COUNT / (CHOICES_PER_QUE + 1)
    return items_per_row   #3


def get_item_interval():
    interval = CHOICES_PER_QUE + 1
    return interval


def delete_rect(cents_pos, que_cnts):
    count = 0
    for i, c in enumerate(cents_pos):
        area_ration = cv2.contourArea(que_cnts[i - count]) / (c[2] * c[3])
        ratio = 1.0 * c[2] / c[3]
        if 0.5 > ratio or ratio > 2 or area_ration < 0.5:
            que_cnts.pop(i - count)
            count += 1
    return que_cnts


def get_left_right(cnts):
    sort_res = contours.sort_contours(cnts, method="top-to-bottom")
    cents_pos = sort_res[1]
    que_cnts = list(sort_res[0])
    que_cnts = delete_rect(cents_pos, que_cnts)

    sort_res = contours.sort_contours(que_cnts, method="top-to-bottom")
    cents_pos = sort_res[1]
    que_cnts = list(sort_res[0])

    num = len(cents_pos) - CHOICE_COL_COUNT + 1
    dt = {}
    for i in range(num):
        distance = 0
        for j in range(i, i + CHOICE_COL_COUNT - 1):
            distance += cents_pos[j + 1][1] - cents_pos[j][1]
        dt[distance] = cents_pos[i:i + CHOICE_COL_COUNT]
    keys = dt.keys()
    key_min = min(keys)
    #if key_min >= 10:
    #    raise
    w = sorted(dt[key_min], key=lambda x: x[0])
    lt, rt = w[0][0] - w[0][2] * 0.5, w[-1][0] + w[-1][2] * 0.5
    count = 0
    for i, c in enumerate(cents_pos):
        if c[0] < lt or c[0] > rt:
            que_cnts.pop(i - count)
            count += 1
    return que_cnts


def get_top_bottom(cnts):
    sort_res = contours.sort_contours(cnts, method="left-to-right")
    cents_pos = sort_res[1]
    que_cnts = list(sort_res[0])
    choice_row_count = get_choice_row_count()
    num = len(cents_pos) - choice_row_count + 1
    dt = {}
    for i in range(num):
        distance = 0
        for j in range(i, i + choice_row_count - 1):
            distance += cents_pos[j + 1][0] - cents_pos[j][0]
        dt[distance] = cents_pos[i:i + choice_row_count]
    keys = dt.keys()
    key_min = min(keys)
    #if key_min >= 10:
    #    raise
    w = sorted(dt[key_min], key=lambda x: x[1])
    top, bottom = w[0][1] - w[0][3] * 0.5, w[-1][1] + w[-1][3] * 0.5
    count = 0
    for i, c in enumerate(cents_pos):
        if c[1] < top or c[1] > bottom:
            que_cnts.pop(i - count)
            count += 1
    return que_cnts


def get_choice_row_count():
    choice_row_count = int(math.ceil(CHOICE_CNT_COUNT * 1.0 / CHOICE_COL_COUNT))
    return choice_row_count


def sort_by_row(cnts_pos):
    choice_row_count = get_choice_row_count()
    count = 0
    rows = []
    threshold = get_min_row_interval(cnts_pos)
    for i in range(choice_row_count):
        cols = cnts_pos[i * CHOICE_COL_COUNT - count:(i + 1) * CHOICE_COL_COUNT - count]
        # threshold = _std_plus_mean(cols)
        temp_row = [cols[0]]
        for j, col in enumerate(cols[1:]):
            if col[1] - cols[j - 1][1] < threshold:
                temp_row.append(col)
            else:
                break
        count += CHOICE_COL_COUNT - len(temp_row)
        temp_row.sort(key=lambda x: x[0])
        rows.append(temp_row)

    # insert_no_full_row(rows)
    ck_full_rows_size(rows)
    return rows


def sort_by_col(cnts_pos):
    # TODO
    cnts_pos.sort(key=lambda x: x[0])
    choice_row_count = get_choice_row_count()
    count = 0
    cols = []
    threshold = get_min_col_interval(cnts_pos)
    for i in range(CHOICE_COL_COUNT):
        rows = cnts_pos[i * choice_row_count - count:(i + 1) * choice_row_count - count]
        temp_col = [rows[0]]
        for j, row in enumerate(rows[1:]):
            if row[0] - rows[j - 1][0] < threshold:
                temp_col.append(row)
            else:
                break
        count += choice_row_count - len(temp_col)
        temp_col.sort(key=lambda x: x[1])
        cols.append(temp_col)
    ck_full_cols_size(cols)
    return cols


def insert_null_2_rows(cols, rows):
    temp = {}
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            try:
                if row[j] != col[0]:
                    row.insert(j, (col[1][0], row[j][1], col[1][2], row[j][3]))
                else:
                    temp[j] = col.pop(0)
            except IndexError:
                try:
                    row.insert(j, (col[1][0], row[j - 1][1], col[1][2], row[j - 1][3]))
                except IndexError:
                    row.insert(j, (temp[j][0], row[j - 1][1], temp[j][2], row[j - 1][3]))


def get_min_row_interval(cnts_pos):
    choice_row_count = get_choice_row_count()
    rows_interval = []
    for i, c in enumerate(cnts_pos[1:]):
        rows_interval.append(c[1] - cnts_pos[i][1])
    rows_interval.sort(reverse=True)
    return min(rows_interval[:choice_row_count - 1])


def get_min_col_interval(cnts_pos):
    cols_interval = []
    for i, c in enumerate(cnts_pos[1:]):
        cols_interval.append(c[0] - cnts_pos[i][0])
    cols_interval.sort(reverse=True)
    return min(cols_interval[:CHOICE_COL_COUNT - 1])


def get_min_interval(cnts_pos, orient):
    idx = ORIENT_CODE[orient]
    interval_list = []


def ck_full_rows_size(rows):
    count = 0
    for row in rows:
        if len(row) == CHOICE_COL_COUNT:
            count += 1
    #if count < 1:
    #   raise


def ck_full_cols_size(rows):
    choice_row_count = get_choice_row_count()
    count = 0
    for row in rows:
        if len(row) == choice_row_count:
            count += 1
    #if count < 1:
    #   raise


def get_vertical_projective(img):
    w = [0] * img.shape[1]
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            t = cv2.cv.Get2D(cv2.cv.fromarray(img), y, x)
            if t[0] == 255:
                w[x] += 1
    # show_fuck(img, w)
    seg(w, img)
    return w


def get_h_projective(img):
    h = [0] * img.shape[0]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            s = cv2.cv.Get2D(cv2.cv.fromarray(img), y, x)
            if s[0] == 255:
                h[y] += 1
    painty = np.zeros(img.shape, np.uint8)
    painty = painty + 255
    for y in range(img.shape[0]):
        for x in range(h[y]):
            cv2.cv.Set2D(cv2.cv.fromarray(painty), y, x, (0, 0, 0, 0))
    cv2.imshow('painty', painty)
    cv2.waitKey(0)


def show_fuck(img, w):
    paintx = np.zeros(img.shape, np.uint8)
    paintx = paintx + 255
    for x in range(img.shape[1]):
        for y in range(w[x]):
            # 把为0的像素变成白
            cv2.cv.Set2D(cv2.cv.fromarray(paintx), y, x, (0, 0, 0, 0))

    # 显示图片
    cv2.imshow('paintx', paintx)
    cv2.waitKey(0)


def seg(w, img):
    dt = Counter(w)
    counts = np.array(dt.values())
    mean = np.mean(counts)
    std = np.std(counts)
    _w = np.array(w)
    _w[_w <= mean + std * 2] =0
    count = 0
    for i, _ in enumerate(_w[1:]):
        if _ > 0 and _w[i] == 0:
            count += 1
    if _w[-1] > 0:
        count -= 1
    print count
    if count != 18:
        show_fuck(img, w)
        show_fuck(img, _w)
        print '1'
    return _w