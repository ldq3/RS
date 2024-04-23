import scipy.sparse as sp
import numpy as np
import random
import torch
from torch_geometric.data import Data

class Bipartite_Graph:
    '''
    Description
    -----------
    The bipartite graph to represent the rating of user and item. 

    You need to pass a DataFrame with columns "user", "item", "rating" to initialize the class. It's recommended to pass the complete dataset rather than the splited dataset, as the class will map the user and item id to a continues [0, N) range.

    The primary function of this class is to extract the enclosing subgraph around the given edge. The subgraph is a bipartite graph with the target edge removed. The subgraph is then converted to the PyG Data object.

    Arg
    ---
    1. table: DataFrame, with columns "user", "item", "rating"
    - id_mapped: bool which default False, whether the user and item id had be mapped to a continues [0, N) range
    - map_rating: function, which default None, map the rating to a continues [1, N) range
    - extracting_setting:
        - h: int, which default 1, the hop of the enclosing subgraph
        - sample_ratio: float, which default 1.0, the ratio of the nodes to be sampled in each hop
        - max_nodes_per_hop: int, which default None, the max number of nodes to be sampled in each hop

    Attribute
    ---------
    - edges: ndarray, the edges of the bipartite graph
    - if you set the id_mapped to False, the class will generate two attributes user_id_dict and item_id_dict:
        - user_id_dict: dict, the mapping from the original user id to the new user id
        - item_id_dict: dict, the mapping from the original item id to the new item id
    - csr: scipy.sparse.csr_matrix, the user-item rating matrix
    - csc: scipy.sparse.csc_matrix, the item-user rating matrix

    Method
    ------
    - extract_data: extract the enclosing subgraph around the given edge and convert the subgraph to the PyG Data object
    - extract_subgraph: extract the enclosing subgraph around the given edge
    - subgraph2data: convert the subgraph to the PyG Data object
    - u_neighbors: get the neighbors of the user nodes
    - i_neighbors: get the neighbors of the item nodes
    '''
    def __init__(
    self,
    table,
    id_mapped = False,
    map_rating = None,
    h = 1,
    sample_ratio = 1.0,
    max_nodes_per_hop = None,
    ) -> None:
        # if the variable id_mapped is False, the class will map user and item to a continues [0, N) range, and generate two attributes user_id_dict and item_id_dict 
        if id_mapped == False:
            table["user"], self.user_id_dict = self.__map_id(table["user"])
            table["item"], self.item_id_dict = self.__map_id(table["item"])
        
        if map_rating is not None:
            table["rating"] = map_rating(table["rating"])
        
        # instance attributes
        self.edges = table[["user", "item"]].values
        '''
        a ndarray with shape n * (user_id, item_id).
        '''
        self.csr = sp.csr_matrix((table["rating"].values, (self.edges[:, 0], self.edges[:, 1])))
        self.csc = self.csr.tocsc()
        self.__h = h
        self.__sample_ratio = sample_ratio
        self.__max_nodes_per_hop = max_nodes_per_hop 
    
    @staticmethod
    def __map_id(id):
        """
        Map data to proper indices in case they are not in a continues [0, N) range

        Arg
        ----------
        data : np.int32 arrays

        Return
        -------
        mapped_data : np.int32 arrays
        n : length of mapped_data

        """
        uniq = list(set(id))

        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        mapped_id = np.array([id_dict[x] for x in id])

        return mapped_id, id_dict
    
    @staticmethod
    def __one_hot(idx, length):
        '''
        one-hot encoding, 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
        '''
        return np.eye(length)[idx]

    def u_neighbors(self, fringe):
        return set(self.csr[list(fringe)].indices) if fringe else set([])
    
    def i_neighbors(self, fringe):
        return set(self.csc[:, list(fringe)].indices) if fringe else set([])

    def extract_data(self, edge):
        '''
        '''
        u_nodes, u_dist, v_nodes, v_dist = self.extract_subgraph(edge)
        return self.subgraph2data(u_nodes, u_dist, v_nodes, v_dist)

    def extract_subgraph(self, edge):
        '''
        extract enclosing subgraph from the bipartite graph around the given edge
        
        Arg
        ---
        1. edge: tuple or list, (user_id, item_id)

        Return
        ------
        u_nodes, u_dist, v_nodes, v_dist
        '''
        u_nodes, v_nodes = [edge[0]], [edge[1]]
        u_dist, v_dist = [0], [0]

        u_visited, v_visited = {edge[0]}, {edge[1]}
        u_fringe, v_fringe = {edge[0]}, {edge[1]}
        for dist in range(1, self.__h+1):
            v_fringe, u_fringe = self.u_neighbors(u_fringe), self.i_neighbors(v_fringe)
            u_fringe = u_fringe - u_visited
            v_fringe = v_fringe - v_visited
            u_visited = u_visited.union(u_fringe)
            v_visited = v_visited.union(v_fringe)
            if self.__sample_ratio < 1.0:
                u_fringe = random.sample(u_fringe, int(self.__sample_ratio*len(u_fringe)))
                v_fringe = random.sample(v_fringe, int(self.__sample_ratio*len(v_fringe)))
            if self.__max_nodes_per_hop is not None:
                if self.__max_nodes_per_hop < len(u_fringe):
                    u_fringe = random.sample(u_fringe, self.__max_nodes_per_hop)
                if self.__max_nodes_per_hop < len(v_fringe):
                    v_fringe = random.sample(v_fringe, self.__max_nodes_per_hop)
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = u_nodes + list(u_fringe)
            v_nodes = v_nodes + list(v_fringe)
            u_dist = u_dist + [dist] * len(u_fringe)
            v_dist = v_dist + [dist] * len(v_fringe)
        return u_nodes, u_dist, v_nodes, v_dist

    def subgraph2data(self, u_nodes, u_dist, v_nodes, v_dist):
        '''
        convert the subgraph to the graph data
        '''
        subgraph = self.csr[u_nodes][:, v_nodes]
        # remove link between target nodes
        y = subgraph[0, 0]
        subgraph[0, 0] = 0
        u, v, r = sp.find(subgraph) 

        # prepare pyg graph constructor inputh
        v += len(u_nodes)
        node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]

        u, v = torch.IntTensor(u), torch.IntTensor(v)
        r = torch.ByteTensor(r)  
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
        edge_type = torch.cat([r, r])
        x = torch.ByteTensor(self.__one_hot(node_labels, 2*self.__h + 2))
        y = torch.ByteTensor([y])

        return Data(x, edge_index, edge_type=edge_type, y=y)