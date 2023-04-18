import math
import sys
import numpy as np
from utils import *

result_path = "./result/sparse_pagerank_block.txt"

data_path = "Data/Data.txt"
sparse_matrix_path = "./output/matrix.txt"
r_old_path = "./output/old.txt"
r_new_path = "./output/new.txt"


# we assume that only 1000 instance can store in RAM
def page_rank_block(node_num, iters=100, beta=0.85, epsilon=1e-6, dist='l1', block_max_num=1000):
    """
    r_old save in r_old_path,
    m save in sparse_matrix_path,
    r_new can't store in RAM, so need to split it into block.
    block num is node_num / 1000
    """
    global r_new_path, r_old_path
    block_num = math.ceil(node_num / block_max_num)
    print("Your n_new split into %d blocks" % block_num)
    iter = 0
    renormalization = 0
    while iter < iters:
        with open(r_new_path, 'w') as f:
            f.truncate()

        r_sum = 0
        for i in range(block_num):
            node_in_block = [(n + i * block_max_num) for n in range(min(block_max_num, node_num - i * block_max_num))]
            # we can store this in RAM, init the r_new as a dict
            v = (1. - beta) / node_num
            r_new = {k: v for k in node_in_block}
            # dot to calculate new rank
            with open(sparse_matrix_path, 'r') as matrix_f, open(r_old_path, 'r') as r_old_f:
                r_index = 0
                for line in matrix_f:
                    m = [int(x) for x in line.split()]
                    s = m[0]
                    while s >= r_index:
                        r_old_s = float(r_old_f.readline()) + renormalization
                        r_index += 1
                    degree = m[1]
                    tos = m[2:]
                    for t in tos:
                        if t not in node_in_block:
                            continue
                        r_new[t] += beta * r_old_s / degree
            r_sum += sum(r_new.values())
            with open(r_new_path, 'a') as f:
                for _, score in r_new.items():
                    f.write("{}\n".format(score))
        # renormailze to solve dead ends
        _renormalization = (1 - r_sum) / node_num
        iter += 1
        if dist == 'L1':
            with open(r_new_path, 'r') as f1, open(r_old_path, 'r') as f2:
                r1 = f1.readlines()
                r2 = f2.readlines()
                e = [abs((float(r1[i]) + _renormalization) - (float(r2[i]) + renormalization)) for i in range(node_num)]
            if sum(e) < epsilon:
                break
        renormalization = _renormalization
        # update r_old
        r_new_path, r_old_path = r_old_path, r_new_path

    print("Total calculate round:", iter)
    return r_new_path


if __name__ == "__main__":
    id_index_dict = load_data(data_path, sparse_matrix_path, r_old_path)
    index_id_dict = {v: k for k, v in id_index_dict.items()}

    r_new_path = page_rank_block(len(id_index_dict), iters=10)
    with open(r_new_path, 'r') as f:
        pr = [float(line.strip()) for line in f]

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
