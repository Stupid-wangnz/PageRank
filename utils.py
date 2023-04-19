import math

import numpy as np
import pandas as pd
from collections import defaultdict


def load_data(data_path, sparse_matrix_path, r_old_path):
    """
    read edges from file
    :return:sparse matrix, and the node set
    """
    nodes = set()
    id_index_dict = {}
    link_matrix = defaultdict(lambda: [0, []])
    with open(data_path) as f:
        for line in f:
            s, t = [int(x) for x in line.split()]
            if s not in nodes:
                id_index_dict[s] = len(nodes)
                nodes.add(s)
            if t not in nodes:
                id_index_dict[t] = len(nodes)
                nodes.add(t)
            s, t = id_index_dict[s], id_index_dict[t]
            link_matrix[s][0] += 1
            link_matrix[s][1].append(t)
    with open(sparse_matrix_path, 'w') as f:
        for s, (degree, ts) in sorted(link_matrix.items(), key=lambda x: x[0]):
            f.write("{} {} {}\n".format(s, degree, ' '.join(str(x) for x in sorted(ts))))
    num = len(nodes)
    with open(r_old_path, 'w') as f:
        for i in range(num):
            f.write("{}\n".format(1. / num))
    return id_index_dict


def break_into_stripes(sparse_matrix_path, node_num, block_max_num=1000):
    """
    need to break link matrix into stripes
    """
    block_num = math.ceil(node_num / block_max_num)
    print("Your n_new split into %d blocks" % block_num)

    stripes_link_matrix = [defaultdict(lambda: [0, []]) for _ in range(block_num)]

    with open(sparse_matrix_path, 'r') as f:
        for line in f:
            x = [int(_) for _ in line.split()]
            s, degree, tos = x[0], x[1], x[2:]
            for t in tos:
                stripes_i = t // block_max_num
                stripes_link_matrix[stripes_i][s][0] = degree
                stripes_link_matrix[stripes_i][s][1].append(t)

    for i in range(block_num):
        stripes_path = sparse_matrix_path + '_' + str(i)
        with open(stripes_path, 'w') as f:
            for s, (degree, ts) in sorted(stripes_link_matrix[i].items(), key=lambda x: x[0]):
                f.write("{} {} {}\n".format(s, degree, ' '.join(str(x) for x in sorted(ts))))


if __name__ == "__main__":
    _ = load_data("Data/Data.txt", "./output/matrix.txt", "./output/old.txt")
    break_into_stripes("./output/matrix.txt", len(_))
