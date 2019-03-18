# -*- coding: utf-8 -*-
# exercise 4.3: 基于信息熵的决策树算法
import numpy as np
import treeplotter
from collections import Counter


def all_class(D):  # 判断D中类的个数
    return len(np.unique(D[:, -1]))  # np.unique用以生成具有不重复元素的数组


def diff_in_attr(D, A_code):  # 判断D中样本在当前A_code上是否存在属性取值不同，即各属性只有一个取值
    cond = False
    for i in A_code[:-2]:  # continuous args are excluded
        if len(np.unique(D[:, i])) != 1:
            cond = True
            break
    return cond


def major_class(D):  # 票选出D中样本数最多的类
    c = Counter(D[:, -1]).most_common()[0][0]
    return c


def max_gain(D, A_code):
    # 对离散属性选取最大信息增益属性
    N = len(D)  # 当前样本数
    dict_class = {}  # 用字典保存类及其所在样本,键为类型，值为类型所含样本
    for i in range(N):
        if D[i, -1] not in dict_class.keys():  # 为字典添加新类
            dict_class[D[i, -1]] = []
            dict_class[D[i, -1]].append(int(D[i, 0]))
        else:
            dict_class[D[i, -1]].append(int(D[i, 0]))
    Gain_D_A = {}  # 用字典保存在某一属性下的属性值及其所在样本,键为属性取值，值为对应样本
    for a in A_code[:-2]:
        # A中的离散属性,在后期的迭代中，属性a可能会变得不连续
        dict_attr = {}
        for i in range(N):
            if D[i, a] not in dict_attr.keys():  # 为字典添加新的属性取值
                dict_attr[D[i, a]] = []
                dict_attr[D[i, a]].append(int(D[i, 0]))
            else:
                dict_attr[D[i, a]].append(int(D[i, 0]))
        # 不用计算真实的Gain(D,a)，因为所有Gain(D,a)的第一项都是Ent(D)，所以直接计算第二项
        # 第a个属性的Gain(D,a)用Gain_D_A[(a,)]表示，键用元组(a,)是为了后面直接用len(key)
        # 判断是离散属性还是连续属性的key，并令初始Gain(D,a)值为0，
        Gain_D_A[(a,)] = 0
        for av, v in dict_attr.items():
            m = len(v)  # m为当前属性取值的样本总数,如A_a0包含的样本总数
            x2 = len(set(v) & set(dict_class['好']))  # 注意考虑x1或x2可能为0
            x1 = m - x2
            print(x1,x2)
            if x1:
                Gain_D_A[(a,)] += x1 * np.log2(x1 / m)
            if x2:
                Gain_D_A[(a,)] += x2 * np.log2(x2 / m)
    for a in A_code[-2:]:
        # A中的连续属性density和sugar
        cmp = {}  # 存放不同划分点对应的Ent
        d_a=[]
        for i in range(N):
            d_a.append(float(D[i, a]))
        sort_d_a = sorted(d_a)
        for t in range(N - 1):
            ls, mr, gain = [], [], 0
            divider = (sort_d_a[t] + sort_d_a[t + 1]) / 2
            for i in range(N):
                if float(D[i, a]) < divider:
                    ls.append(int(D[i, 0]))
                else:
                    mr.append(int(D[i, 0]))
            less, more = len(ls), len(mr)
            less0 = len(set(ls) & set(dict_class['坏']))
            more0 = len(set(mr) & set(dict_class['坏']))
            less1, more1 = less - less0, more - more0
            for p in [less0, less1]:
                if p:
                    gain += p * np.log2(p / less)
            for p in [more0, more1]:
                if p:
                    gain += p * np.log2(p / more)
            cmp[t] = gain
        best_t = int(sorted(cmp, key=lambda x: cmp[x], reverse=True)[0])
        threshold = (sort_d_a[best_t] + sort_d_a[best_t + 1]) / 2
        Gain_D_A[(a, threshold)] = cmp[best_t]
    Gain_D_A_list = sorted(Gain_D_A.items(), key=lambda a: (a[1], -len(a[0])), reverse=True)
    # 如果多个属性同时达到最大信息熵，则优先选取离散属性，即离散属性排在连续属性之前
    best = Gain_D_A_list[0][0]
    print(len(best))
    if len(best) == 2:  # 最优属性为连续属性
        a, threshold = best
        low = [int(D[i, 0]) for i in range(N) if float(D[i, a]) <= threshold]
        high = [int(D[i, 0]) for i in range(N) if float(D[i, a]) > threshold]
        # 返回对应的属性序号（绝对序号），并返回此属性下对应的取值和所包含的实例的id
        return a, {'<=%.4f' % threshold: low, '>%.4f' % threshold: high}
    else:  # 最优属性为离散属性
        dict_attr = {}
        best = int(best[0])
        for i in range(N):
            if D[i, best] not in dict_attr.keys():
                dict_attr[D[i, best]] = []
                dict_attr[D[i, best]].append(int(D[i, 0]))
            else:
                dict_attr[D[i, best]].append(int(D[i, 0]))
        print(dict_attr)
        # 返回对应的属性序号（绝对序号），并返回此属性下对应的取值和所包含的实例的id
        return best, dict_attr


def Tree_Generate(D, A_code, full_D):
    if all_class(D) == 1:  # case1
        return D[0, -1]
    if (len(A_code) == 2) or (not diff_in_attr(D, A_code)):  # case2
        return str(major_class(D))
    a, di = max_gain(D, A_code)
    tree = {A[a]: {}}
    new_A_code = A_code[:]
    if a not in A_code[-2:]:
        all_a = np.unique(full_D[:, a])
        new_A_code.remove(a)
        for item in all_a:
            if item not in di.keys():
                di[item] = []
    for av, Dv in di.items():
        if Dv:
            tree[A[a]][av] = Tree_Generate(full_D[Dv, :], new_A_code, full_D)
        else:  # case3
            tree[A[a]][av] = 'empty: %s' % major_class(D)
    return tree


if __name__ == '__main__':
    D = np.array([[0, '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好'],
                  [1, '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好'],
                  [2, '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好'],
                  [3, '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好'],
                  [4, '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好'],
                  [5, '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好'],
                  [6, '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好'],
                  [7, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好'],
                  [8, '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏'],
                  [9, '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏'],
                  [10, '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏'],
                  [11, '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏'],
                  [12, '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏'],
                  [13, '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏'],
                  [14, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏'],
                  [15, '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏'],
                  [16, '青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏']])
    A = {0: '编号', 1: '颜色', 2: '根蒂', 3: '敲声', 4: '纹理',
         5: '脐部', 6: '触感', 7: '密度', 8: '含糖量', 9: '好瓜'}
    A_code = list(range(1, len(A) - 1))  # A_code = [1, 2, 3, 4, 5, 6, 7, 8]
    print(A)
    tree = Tree_Generate(D, A_code, D)
    print(tree)
    treeplotter.createPlot(tree)
