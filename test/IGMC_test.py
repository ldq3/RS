import sys
sys.path.append("..")
from script.IGMC import *

import numpy as np
import scipy.sparse as sp
import pandas as pd

# test Bipartite_Graph
table = {
"user": [1, 1, 2, 2, 3, 3, 2],
"item": [1, 2, 1, 3, 2, 3, 2],
"rating": [2, 4, 3, 1, 4, 0, 4]
}
input = pd.DataFrame(table)
map_rating = lambda x: x + 1
G = Bipartite_Graph(input, id_mapped=False, map_rating=map_rating)

def test_init():
    edges = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],
        [2, 2],
        [1, 1]
    ])

    rating = np.array([3, 5, 4, 2, 5, 1, 5])

    result = G.edges
    print("\n")
    print("Result:", result)
    print("Target:", edges)
    assert result.all() == edges.all()

    csr = sp.csr_matrix((rating, (edges[:, 0], edges[:, 1])))
    assert G.csr[0, 1] == csr[0, 1]

def test_u_neighbors():
    target = {1, 2}
    result = G.u_neighbors({2})

    print("\n")
    print("Result:", result)
    print("Target:", target)

    assert target == result

def test_i_neighbors():
    target = {1, 2}
    result = G.i_neighbors({2})

    print("\n")
    print("Result:", result)
    print("Target:", target)

    assert target == result

def test_extract_subgraph():
    u_nodes = [2, 1]
    u_dist = [0, 1]
    v_nodes = [2, 1]
    v_dist = [0, 1]
    target = (u_nodes, u_dist, v_nodes, v_dist)
    input = np.array([2, 2])
    result = G.extract_subgraph(input)
    print("\n")
    print("Result:", result)
    print("Target:", target)

    assert target == result

def test_subgraph2data():
    u_nodes = [2, 1]
    u_dist = [0, 1]
    v_nodes = [2, 1]
    v_dist = [0, 1]
    input = (u_nodes, u_dist, v_nodes, v_dist)

    x = torch.ByteTensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
        ]) # 距离的独热编码
    edge_index = torch.IntTensor([
        [0, 1, 1, 3, 2, 3],
        [3, 2, 3, 0, 1, 1]
    ])
    edge_type = torch.ByteTensor([
        5, 2, 5, 5, 2, 5
    ])
    y = torch.ByteTensor([
        1
    ])

    target = Data(x, edge_index, edge_type = edge_type, y = y)

    result = G.subgraph2data(*input)

    print("\n")
    print("Result:", result.y)
    print("Target:", target.y)

    assert target.y == result.y