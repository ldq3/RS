import scipy.sparse as sp
import numpy as np

class Bipartite_Graph:
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
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop 
    
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