import pandas as pd
import numpy as np
import networkx as nx

data_path = "./Data.txt"
N = 8297


def nx_pagerank():
    # generate Graph
    G = nx.DiGraph()
    with open(data_path) as f:
        for line in f:
            s, t = [int(x) for x in line.split()]
            G.add_edge(s, t)
    pr = nx.pagerank(G)
    with open('./result/nx_package.txt', 'w') as f:
        for node, value in sorted(pr.items(), key=lambda x: x[1], reverse=True):
            f.write("{}\t{}\n".format(node, value))


def load_data(file):
    f = open(data_path, 'r')
    edges = []
    for line in f:
        edges.append([int(x) for x in line.split()])
    G = np.zeros([N, N])
    for edge in edges:
        G[edge[0] - 1, edge[1] - 1] = 1

    pr = np.ones(N) / N
    return G, pr


def pagerank_easy(alpha=0.85, tol=1e-6, max_iteration=100):
    G, pr = load_data(data_path)
    for j in range(N):
        sum_of_col = sum(G[:, j])
        if sum_of_col > 0:
            for i in range(N):
                G[i, j] /= sum_of_col
    print(G)
    links = alpha * G + (1 - alpha) / N * np.ones([N, N])

    iterations = 0
    while True:
        pr_new = np.dot(links, pr)
        e = pr_new - pr
        e = max(map(abs, e))
        pr = pr_new
        if iterations % 10 == 0:
            print(iterations, pr)
        if e <= tol or iterations >= max_iteration:
            break
        iterations += 1
    return pr


if __name__ == "__main__":
    pr = pagerank_easy()
    print(pr)
