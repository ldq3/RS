import scipy.sparse as sp
import numpy as np

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