import numpy as np
import read_file as rf
import perform_evaluate as pe


# user_p_i, item_ped_u, i_te_u, u_test
user_p_i, item_ped_u, i_te_u, u_test, yui = rf.read_train_test()

n = 943
m = 1682
K = 50

Skj = np.zeros((m, m), dtype=np.float)
max_skj = np.zeros(m, dtype=np.float)

# 计算相似度 Skj
for item_k in range(m):
    for item_j in range(m):
        uk = item_ped_u[item_k]
        uj = item_ped_u[item_j]

        numerator = len(list(set(uk) & set(uj)))
        denominator = len(list(set(uk) | set(uj)))

        if denominator != 0:
            Skj[item_k][item_j] = float(numerator) / denominator
        if max_skj[item_k] < Skj[item_k][item_j]:
            max_skj[item_k] = Skj[item_k][item_j]

# 正则化
for item_k in range(m):
    for item_j in range(m):
        if max_skj[item_k] != 0:
            Skj[item_k][item_j] /= max_skj[item_k]

# for item j  key item  value list the nearest K items of item j
N_j = {}
for item_j in range(m):
    N_j.setdefault(item_j, [])

    tmp_dic = {}
    for item in range(m):
        if item_j != item:
            tmp_dic[item] = Skj[item_j][item]
    sorted_dic = sorted(tmp_dic.items(), key=lambda x: x[1], reverse=True)

    for k in range(K):
        N_j[item_j].append(sorted_dic[k][0])

# start item-based one-class collaborative filtering
predict_ruj = np.zeros((n, m), dtype=float)
i_re_u = {}

K = 5
for u in u_test:
    tmp_dic = {}
    i_re_u[u] = []
    for j in range(m):
        # !!!!!!!
        '''
            一直没有发现这个错误，浪费了好多时间 就是少了这两句：
                if yui[u][j] == 1:      # 训练集中的数据， yui[u][j] == 1
                    continue            # 不需要进行predict rating
            正因为没有这两句，使得 intersection 混入了许多错误数据，
            从而使得 sorted_dic 排序正确，但是前K 个中却有了错误数据 
            ！！！！！！！！！！！！！！！！！！！！！！！！！
        '''
        # !!!!!!!
        if yui[u][j] == 1:
            continue
        i_u = user_p_i[u]
        n_j = N_j[j]
        intersection = list(set(i_u) & set(n_j))
        tmp = 0.0
        for k in intersection:
            tmp += Skj[k][j]
        predict_ruj[u][j] = tmp
        tmp_dic[j] = tmp
    sorted_dic = sorted(tmp_dic.items(), key=lambda x: x[1], reverse=True)
    for k in range(K):
        i_re_u[u].append(sorted_dic[k][0])

#  Evaluation
pre_k, rec_k = pe.get_pre_rec(i_re_u, i_te_u)
print('Pre@5:', round(pre_k, 4))
print('Rec@5:', round(rec_k, 4))
