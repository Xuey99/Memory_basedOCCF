
def get_pre_rec(i_re_u, i_te_u):
    k = 5
    pre_k = 0.0
    rec_k = 0.0

    u_test_num = len(i_te_u)
    pre_ku = {}
    rec_ku = {}
    for u in i_re_u:
        i_re = i_re_u[u]
        i_te = i_te_u[u]
        num = len(list(set(i_re) & set(i_te)))
        i_te_u_len = len(i_te)
        pre_ku[u] = num / k
        rec_ku[u] = num / i_te_u_len
        pre_k += num / k
        rec_k += num / i_te_u_len
    pre_k = pre_k / u_test_num
    rec_k = rec_k / u_test_num

    return pre_k, rec_k
