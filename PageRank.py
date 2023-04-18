import math
import sys
import numpy as np
from utils import *

result_path = "./result/sparse_pagerank_base.txt"

data_path = "Data/Data.txt"
sparse_matrix_path = "./output/matrix.txt"
r_old_path = "./output/old.txt"


def euclidean_distance(r_new):
    """
    use Euclidean distance to calculate when to stop
    :param r_new:
    :return:e_dis
    """
    with open(r_old_path, 'r') as f:
        r_old = f.readlines()
        dist = [(r_new[i] - float(r_old[i])) ** 2 for i in range(len(r_new))]
    dist = math.sqrt(sum(dist))
    return dist


def l1_norm(r_new):
    with open(r_old_path, 'r') as f:
        r_old = f.readlines()
        dist = [abs(r_new[i] - float(r_old[i])) for i in range(len(r_new))]
    dist = sum(dist)
    return dist


def page_rank(node_num, iters=100, beta=0.85, epsilon=1e-6, dist='l1'):
    """
    in RAM only have r_new
    r_old save in r_old_path
    m save in sparse_matrix_path

    :param node_num:
    :param iters:
    :param beta:
    :param epsilon:
    :param dist: l1_norm or euclidean
    :return:pr
    """
    iter = 0
    while iter < iters:
        r_new = [(1. - beta) / node_num] * node_num
        # dot to calculate new rank
        with open(r_old_path, 'r') as r_old_f:
            r_index = 0
            with open(sparse_matrix_path, 'r') as f:
                for line in f:
                    m = [int(x) for x in line.split()]
                    s = m[0]
                    while s != r_index:
                        r_old_s = r_old_f.readline()
                        r_index += 1
                    r_old_s = float(r_old_f.readline())
                    r_index += 1
                    degree = m[1]
                    tos = m[2:]
                    for t in tos:
                        r_new[t] += beta * r_old_s / degree
        # renormailze to solve dead ends
        r_new = [(i + (1 - float(sum(r_new))) / node_num) for i in r_new]
        iter += 1
        if dist == 'euclidean':
            if euclidean_distance(r_new) < epsilon:
                break
        elif dist == 'l1':
            if l1_norm(r_new) < epsilon:
                break
        # update r_old
        with open(r_old_path, 'w') as f:
            for i in range(node_num):
                f.write("{}\n".format(r_new[i]))

    print("Total calculate round:", iter)
    return r_new


if __name__ == "__main__":
    id_index_dict = load_data(data_path, sparse_matrix_path, r_old_path)
    index_id_dict = {v: k for k, v in id_index_dict.items()}

    pr = page_rank(len(id_index_dict), iters=10)

    id_rank = {}
    for index, v in enumerate(pr):
        id_rank[index_id_dict[index]] = v
    with open(result_path, 'w') as f:
        num = 0
        for id, value in sorted(id_rank.items(), key=lambda x: x[1], reverse=True):
            f.write("{}\t{}\n".format(id, value))
            num += 1
            if num == 100:
                break
