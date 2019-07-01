import random

import numpy as np
from scipy.sparse import coo_matrix
from game import defines



class Crossings:
    def __init__(self,tiles,harbours):
        """
        Intializes the crossing variables.

        """
        # Indices of neighbouring crossings
        self.neighbouring_crossings=[[1, 8], [2, 0], [3, 1, 10], [4, 2], [5, 3, 12], [6, 4], [5, 14],
                                     [8,17],[0,7,9],[8,10,19],[2,9,11],[10,12,21],[4,11,13],[12,14,23],[13,15,6],[14,25],
                                     [17,27],[16,7,18],[17,29,19],[18,9,20],[19,31,21],[20,11,22],[21,23,33],[22,13,24],[23,25,35],[24,15,26],[25,37],
                                     [16,28],[27,29,38],[28,18,30],[29,31,40],[30,20,32],[31,33,42],[32,22,34],[33,35,44],[34,36,24],[35,37,46],[26,36],
                                     [28,39],[38,40,47],[39,30,41],[40,42,49],[41,32,43],[42,44,51],[43,34,45],[44,53,46],[36,45],
                                     [39,48],[47,49],[48,41,50],[49,51],[50,43,52],[51,53],[52,45]]
        # Indices of neighbouring tiles
        self.neighbouring_tiles=[[0],[0],[0,1],[1],[1,2],[2],[2],
                                 [3],[0,3],[0,3,4],[0,1,4],[1,4,5],[1,2,5],[2,5,6],[2,6],[6],
                                 [7],[3,7],[3,7,8],[3,4,8],[4,8,9],[4,5,9],[5,9,10],[5,6,10],[6,10,11],[6,11],[11],
                                 [7],[7,12],[7,12,8],[8,12,13],[8,9,13],[13,9,14],[9,10,14],[10,14,15],[10,11,15],[11,15],[11],
                                 [12],[12,16],[12,16,13],[13,16,17],[13,14,17],[14,17,18],[14,15,18],[15,18],[15],
                                 [16],[16],[16,17],[17],[17,18],[18],[18]]
        connected_tiles = []

        # Combines the connected tile information (resource type, number) to each crossing
        for neighbouring_tiles in self.neighbouring_tiles:
            _tiles=[]
            for tile in neighbouring_tiles:
                tile_list = list(tiles[tile])
                tile_list.append(tile)
                _tiles.append(tile_list)
            connected_tiles.append(_tiles)

        # Crossing indexes where a harbour is located
        harbours_ind=[[0,1],[3,4],[6,15],[26,37],[45,53],[51,52],[47,48],[27,28],[7,17]]
        harbours_lst =[0]*len(self.neighbouring_crossings)

        # Sets the harbour type for each crossing
        for h,t in zip(harbours_ind,harbours):
            for ele in h:
                harbours_lst[ele] = t
        self.crossings = list(zip(self.neighbouring_crossings,connected_tiles,harbours_lst))
        self.building_state = np.array([0]*len(connected_tiles))

        self.crossings_per_tile = []
        self.create_crossings_per_tile()


    def place_settlement(self,crossing_index,player_num):
        """
        Places a settlement on the respective crossing and sets the state of the surrounding
        crossings to 9 (not possible to build here anymore)

        :param crossing_index:
			Index on which the settlement is placed

			    player_num:
			Player number

        """
        self.building_state[crossing_index] = player_num
        self.building_state[self.neighbouring_crossings[crossing_index]] = 9


    def get_crossings(self):
        """
        Returns the crossing list containing all relevant information about the crossings.

        :return:
            list(tuple(neighbour_crossing_indices, neighbouring_tiles,harbours), length 54 (amount of crossings)
                Example of one list element: ([1, 8], [(1, 12)], 3) ... :
                Neighbouring Crossings indices: 1 and 8 , Neighbouring Tiles: Fields with number 12 (only 1 tile),
                Harbour: Ore 2:1
        """
        return self.crossings

    def get_neighbouring_crossings(self):
        """
        Returns neighbouring_crossings

        :return:
            list(list(int))
                Indices of the crossings adjacent to each crossing.
        """
        return self.neighbouring_crossings

    def get_neighbouring_tiles(self):
        """
        Returns neighbouring_tiles

        :return:
            list(list(int))
                Indices of the tiles adjacent to each crossing.
        """
        return self.neighbouring_tiles

    def get_building_state(self):
        """
        Returns relevant state information about the crossings, namely what building is placed at each crossing.

        :return:
            list(int) (length 54)
                Building type: 0 - no building, 1 - Settlement P1, 2 - Settlement P2, 3 - Settlement P3, 4 - Settlement P4
                5 - City P1, 6 - City P2, 7 - City P3, 8 - City P4, 9 - No building possible
        """
        return self.building_state

    def create_connected_roads(self,crossings_connected_to_road):
        """
        Creates the class variable connected_roads containing the edge indices of edges connected to this crossing.

        :param crossings_connected_to_road:
			Vector of crossings chich are connected to each road

        """
        self.connected_roads =[[] for i in range(defines.NUM_CROSSINGS)]
        for i in range(len(crossings_connected_to_road)):
            self.connected_roads[crossings_connected_to_road[i,0]].append(i)
            self.connected_roads[crossings_connected_to_road[i,1]].append(i)


    def get_connected_roads(self):
        """
        Returns the class variable connected_roads containing the edge indices of edges connected to this crossing.

        :return connected_road:
			Vector of list of edges connected to each crossing
        """
        return self.connected_roads

    def place_city(self,crossing_index,player_num):
        """
        Places a city on the respective crossing.

        :param crossing_index:
			Index on which the settlement is placed

			    player_num:
			Player number

        """
        self.building_state[crossing_index] = player_num + 4

    def create_crossings_per_tile(self):
        """
        Creates the 6 crossing indices for each tile out of the neighbouring_tiles variable
        """
        self.crossings_per_tile = [[] for i in range(defines.NUM_TILES)]
        for j in range(len(self.neighbouring_tiles)):
            for tiles in self.neighbouring_tiles[j]:
                self.crossings_per_tile[tiles].append(j)
        self.crossings_per_tile = np.array(self.crossings_per_tile)

    def get_crossings_per_tile(self):
        """
        Returns the class variable crossings_per_tile containing all 6 crossing indices for each tile.

        :return crossings_per_tile:
			2D numpy array tile_index x crossing_index
        """
        return self.crossings_per_tile
