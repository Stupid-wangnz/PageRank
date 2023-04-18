import math
import sys
import numpy as np
from utils import *

result_path = "./result/sparse_pagerank_base.txt"


def euclidean_distance(r_old, r_new):
    """
    use Euclidean distance to calculate when to stop
    :param r_old:
    :param r_new:
    :return:e_dis
    """
    dist = [(r_new[i] - r_old[i]) ** 2 for i in range(len(r_old))]
    dist = math.sqrt(sum(dist))
    return dist


def page_rank(node_num, iters=1000, beta=0.8, epsilon=1e-6):
    r_init = [1. / node_num] * node_num

    r_old = r_init
    r_new = r_old
    iter = 0
    while iter < iters:
        r_new = [(1. - beta) / node_num] * node_num
        # dot to calculate new rank
        with open(sparse_matrix_path, 'r') as f:
            for line in f:
                m = [int(x) for x in line.split()]
                s = m[0]
                degree = m[1]
                tos = m[2:]
                for t in tos:
                    r_new[t] += beta * r_old[s] / degree
        # renormailze to solve dead ends
        r_new = [(i + (1 - float(sum(r_new))) / node_num) for i in r_new]
        if euclidean_distance(r_new, r_old) < epsilon:
            break
        r_old = r_new
        iter += 1

    print("Total calculate round:", iter)
    return r_new


if __name__ == "__main__":
    id_index_dict = load_data()
    index_id_dict = {v: k for k, v in id_index_dict.items()}

    pr = page_rank(len(id_index_dict))

    dt = {}
    for index, v in enumerate(pr):
        dt[index_id_dict[index]] = v
    with open(result_path, 'w') as f:
        num = 0
        for id, value in sorted(dt.items(), key=lambda x: x[1], reverse=True):
            f.write("{}\t{}\n".format(id, value))
            num += 1
            if num == 100:
                break
