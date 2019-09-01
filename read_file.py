import numpy as np


def read_train_test():
    n = 943
    m = 1682

    # users in test set
    u_test = []

    # dict  key for every user(u) in the training set, value for items preferred by user u
    user_p_i = {}

    # dict  key for every item(i) in the training set, value for users preferring i
    item_ped_u = {}

    i_te_u = {}

    yui = np.zeros((n, m), dtype=float)

    for u in range(n):
        user_p_i.setdefault(u, [])

    for i in range(m):
        item_ped_u.setdefault(i, [])

    with open('u1.base', 'r') as train_file:
        for line in train_file.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            if r >= 4:
                user_p_i[uid - 1].append(iid - 1)
                item_ped_u[iid - 1].append(uid - 1)
                yui[uid - 1][iid - 1] = 1

    with open('u1.test', 'r') as test_file:
        for line in test_file.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            if r >= 4:
                i_te_u.setdefault(uid - 1, [])
                i_te_u[uid - 1].append(iid - 1)
                if uid - 1 not in u_test:
                    u_test.append(uid - 1)

    return user_p_i, item_ped_u, i_te_u, u_test, yui
