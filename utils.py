import numpy as np
import pandas as pd
from collections import defaultdict

data_path = "./Data.txt"
sparse_matrix_path = "./output/matrix.txt"


def load_data():
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
        for s, (degree, ts) in link_matrix.items():
            f.write("{} {} {}\n".format(s, degree, ' '.join(str(x) for x in sorted(ts))))
    return id_index_dict


if __name__ == "__main__":
    nodes = load_data()
    print(len(nodes))
