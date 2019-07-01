import random

import numpy as np
from scipy.sparse import coo_matrix

from game import defines

class Roads:
    def __init__(self,neighbouring_crossings):
        """
        Intializes the roads variables.

        """
        roads = []
        for i in range(len(neighbouring_crossings)):
            for ele in neighbouring_crossings[i]:
                conn = [i,ele]
                conn.sort()
                roads.append(conn)
        roads = list(set(tuple(road) for road in roads))
        roads.sort()
        self.roads = roads
        self.road_state = [0]*len(roads)

        self.connected_roads = []
        self.create_connected_roads()

    def get_roads(self):
        """
        Return information about the road

        :return:
            list(tuple(int,int)) (length 72 - number of road positions)
                Each tuple describes the two crossings indices between which the road is situated.
                Road type: 0 - no road, 1 - Road P1, 2 - Road P2, 3 - Road P3, 4 - Road P4

        """
        return np.array(self.roads)

    def get_state(self):
        """
        Returns relevant state information about the roads, namely what road is placed at each crossing.

        :return:
            list(int) (length 72)
                Road type: 0 - no road, 1 - Road P1, 2 - Road P2, 3 - Road P3, 4 - Road P4
        """
        return np.array(self.road_state)

    def create_connected_roads(self):
        """
        Creates connected_roads list out of the crossings each road is connected to.

        """
        self.connected_roads = [[] for i in range(72)]

        for i in range(len(self.roads)):
            for j in range(len(self.roads)):
                if i==j:
                    continue
                if len(set(self.roads[i]).intersection(set(self.roads[j])))>0:
                    self.connected_roads[i].append(j)

    def get_connected_roads(self):
        """
        Returns connected road indices of each road index in a list

        :return:
            list(list[int]) (length 72)
                Each list[int] contains indices of roads connected to this road
        """
        return self.connected_roads



    def place_road(self,road_index,player_num):
        self.road_state[road_index] = player_num