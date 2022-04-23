import numpy as np
import xgboost as xgb
from utils.loss_functions import SquareLoss
"""
    通过Adaboost, GBDT, XGBoost解决二类回归还有多分类问题
    二类回归问题：
        已知如表所示的训练数据，x的取值范围为区间[0.5,10.5]，y的取值范围为区间[5.0,10.0]
        学习这个回归问题的提升树模型，考虑只用树桩作为基函数
        训练数据集
        --------------------------------------------------------------
        x_i     1       2       3       4       5       6       7       8       9       10
        y_i     5.56    5.70    5.91    6.40    6.80    7.07    8.90    8.70    9.00    9.05

    多分类问题：
        
"""

class Sp(object):
    def __init__(self):
        self.data = None
        self.c1 = None
        self.c2 = None


def get_c(N, r):
    c = 0
    for i in range(N):
        c += r[i]
    c /= N
    return c


def get_loss_1(c, N, r):
    val = 0
    for i in range(N):
        val += pow((r[i] - c), 2)
    return val


def get_loss_2(res):
    val = 0
    for i in range(len(res)):
        val += pow(res[i], 2)
    return val


def get_f(x, sp_set):
    f = 0
    sp_set = sorted(sp_set, key=lambda sp: sp.data)

    for sp in sp_set:
        if x < sp.data:
            f += sp.c1
        else:
            f += sp.c2
    return f


def get_residual_table_1(r1, r2, N1, N2, sp_set):
    res = []
    # print(N1)
    for i in range(N1):
        res.append(r1[i] - get_f(i, sp_set))
    for i in range(N2):
        res.append(r2[i] - get_f(i + N1 + 1, sp_set))
    return res


def get_gradient(r1, r2, N1, N2, sp_set):
    res = []
    # print(N1)
    for i in range(N1):
        res.append(SquareLoss().gradient(r1[i], get_f(i, sp_set)))
    for i in range(N2):
        res.append(SquareLoss().gradient(r2[i], get_f(i + N1 + 1, sp_set)))
    return res


def get_split_point(label, split):
    min = float('inf')
    idx = -1
    for _, s in enumerate(split):  # 获得初始分割点s
        # tmp = label  # 切记不要将列表直接赋值给一个列表，这样并不能得到两个列表
        # tmp = label[:]  # 列表副本正确创建方法，使用切片复制列表副本
        r1 = label[0:int(s)]
        r2 = label[int(s):]
        N1 = len(r1)
        N2 = len(r2)
        c1 = get_c(N1, r1)
        c2 = get_c(N2, r2)
        loss = get_loss_1(c1, N1, r1) + get_loss_1(c2, N2, r2)
        if loss < min:
            min = loss
            idx = int(s)
    r1 = label[0:idx]
    r2 = label[idx:]
    # print(r1)
    # print(r2)
    N1 = len(r1)
    N2 = len(r2)
    c1 = get_c(N1, r1)
    c2 = get_c(N2, r2)
    sp = Sp()
    sp.data = split[idx-1]
    sp.c1 = c1
    sp.c2 = c2
    return idx, sp


def adaboost_regression(label, split):
    cnt = 1
    loss = 6
    sp_set = []  # 最佳割点集
    residual = label[:]
    while loss > 0.12:
        cnt += 1
        idx, sp = get_split_point(residual, split)
        sp_set.append(sp)
        r1 = label[0:idx]
        r2 = label[idx:]
        N1 = len(r1)
        N2 = len(r2)
        residual = get_residual_table_1(r1, r2, N1, N2, sp_set)
        loss = get_loss_2(residual)
        print('loss: ', loss)


def gdbt_regression():
    cnt = 1
    loss = 6
    sp_set = []  # 最佳割点集
    residual = label[:]
    while loss > 0.12:

        cnt += 1

        idx, sp = get_split_point(residual, split)

        sp_set.append(sp)

        r1 = label[0:idx]
        r2 = label[idx:]

        N1 = len(r1)
        N2 = len(r2)
        residual = get_gradient(r1, r2, N1, N2, sp_set)

        loss = get_loss_2(residual)
        print('loss: ', loss)


def xgboost_regression():
    pass


def adaboost_multi_classification():
    pass


def gdbt_multi_classification():
    pass


def xgboost_multi_classification():
    pass


if __name__ == '__main__':
    # datasets_1
    label = [5.56, 5.70, 5.91, 6.40, 6.80, 7.07, 8.90, 8.70, 9.00, 9.05]
    split = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

    adaboost_regression(label, split)
    # gdbt_regression(label, split)
