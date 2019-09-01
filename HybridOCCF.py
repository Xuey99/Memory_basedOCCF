import numpy as np
import read_file as rf
import perform_evaluate as pe

# user_p_i, item_ped_u, i_te_u, u_test, yui
user_p_i, item_ped_u, i_te_u, u_test, yui = rf.read_train_test()

n = 943
m = 1682
K = 50

Swu = np.zeros((n, n), dtype=np.float)
max_swu = np.zeros(n, dtype=np.float)

# 计算相似度 Swu
for user_w in range(n):
    for user_u in range(n):
        iw = user_p_i[user_w]
        iu = user_p_i[user_u]

        numerator = len(list(set(iw) & set(iu)))
        denominator = len(list(set(iw) | set(iu)))

        if denominator != 0:
            Swu[user_w][user_u] = float(numerator) / denominator
        if max_swu[user_w] < Swu[user_w][user_u]:
            max_swu[user_w] = Swu[user_w][user_u]

# 正则化
for user_w in range(n):
    for user_u in range(n):
        if max_swu[user_w] != 0:
            Swu[user_w][user_u] /= max_swu[user_w]

# for user u,  key-user, value-list: the nearest K users of user u
N_u = {}
for user_u in range(n):
    N_u.setdefault(user_u, [])

    tmp_dic = {}
    for user in range(n):
        if user_u != user:
            tmp_dic[user] = Swu[user_u][user]
    sorted_dic = sorted(tmp_dic.items(), key=lambda x: x[1], reverse=True)

    for k in range(K):
        N_u[user_u].append(sorted_dic[k][0])

# start user-based one-class collaborative filtering
predict_ruj = np.zeros((n, m), dtype=float)

K = 5
for u in u_test:
    for j in range(m):
        if yui[u][j] == 1:
            continue
        uj = item_ped_u[j]
        nu = N_u[u]
        intersection = list(set(uj) & set(nu))
        tmp = 0.0
        for w in intersection:
            tmp += Swu[w][u]

        predict_ruj[u][j] = tmp


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
i_re_u = {}

K = 5
for u in u_test:
    tmp_dic = {}
    i_re_u[u] = []
    for j in range(m):
        if yui[u][j] == 1:
            continue
        i_u = user_p_i[u]
        n_j = N_j[j]
        intersection = list(set(i_u) & set(n_j))
        tmp = 0.0
        for k in intersection:
            tmp += Skj[k][j]
        predict_ruj[u][j] += tmp
        tmp_dic[j] = predict_ruj[u][j]
    sorted_dic = sorted(tmp_dic.items(), key=lambda x: x[1], reverse=True)
    for k in range(K):
        i_re_u[u].append(sorted_dic[k][0])

#  Evaluation
pre_k, rec_k = pe.get_pre_rec(i_re_u, i_te_u)
print('Pre@5:', round(pre_k, 4))
print('Rec@5:', round(rec_k, 4))
