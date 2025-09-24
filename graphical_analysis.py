
import numpy as np


class Graphical_Analysis:

    def __init__(self,_path, _frame, _cutoff=2, _max_path = 40):
        self.cutoff = _cutoff
        self.max_path = _max_path
        self.path = _path
        self.frame = _frame


    def create_graph(self, distances):
        tf = distances > self.cutoff
        count = 0

        total = np.sum(tf)
        self.coord_to_index = np.zeros((len(tf), len(tf[0]), len(tf[0][0])), dtype=int) - 1
        self.index_to_coord = []
        self.endpoints = []
        self.edges = []
        for z in range(total):
            self.edges.append([])

        for i in range(len(tf)):
            for j in range(len(tf[i])):
                for k in range(len(tf[i][j])):
                    if not tf[i][j][k]: continue
                    self.coord_to_index[i, j, k] = count
                    self.index_to_coord.append([i, j, k])


                    if i > 0:
                        if tf[i - 1][j][k] and self.coord_to_index[i - 1, j, k] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i - 1, j, k])
                            self.edges[int(self.coord_to_index[i - 1, j, k])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)

                    if i < len(tf) - 1:
                        if tf[i + 1][j][k] and self.coord_to_index[i + 1, j, k] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i + 1, j, k])
                            self.edges[int(self.coord_to_index[i + 1, j, k])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)


                    if j > 0:
                        if tf[i][j - 1][k] and self.coord_to_index[i, j - 1, k] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i, j - 1, k])
                            self.edges[int(self.coord_to_index[i, j - 1, k])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)

                    if j < len(tf[i]) - 1:
                        if tf[i][j + 1][k] and self.coord_to_index[i, j + 1, k] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i, j + 1, k])
                            self.edges[int(self.coord_to_index[i, j + 1, k])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)
                    if k > 0:
                        if tf[i][j][k - 1] and self.coord_to_index[i, j, k - 1] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i, j, k - 1])
                            self.edges[int(self.coord_to_index[i, j, k - 1])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)

                    if k < len(tf[i][j]) - 1:
                        if tf[i][j][k + 1] and self.coord_to_index[i, j, k + 1] != -1:
                            self.edges[int(self.coord_to_index[i, j, k])].append(self.coord_to_index[i, j, k + 1])
                            self.edges[int(self.coord_to_index[i, j, k + 1])].append(self.coord_to_index[i, j, k])
                    else:
                        if count not in self.endpoints:
                            self.endpoints.append(count)

                    count += 1
        self.coord_to_index = np.array(self.coord_to_index)
        self.index_to_coord = np.array(self.index_to_coord)
        self.endpoints = np.array(self.endpoints)

    def CreateFastestPath(self, node, curr_dist):
        if node in self.endpoints:
            return 0
        if self.fastestPath[node] != -1:
            return self.fastestPath[node]
        if curr_dist > self.max_path-1:
            return -1

        self.currentPath[curr_dist] = node
        fastest   = -1
        for potentialEdge in self.edges[node]:
            if potentialEdge in self.currentPath[:curr_dist]:
                continue

            response = self.CreateFastestPath(potentialEdge, curr_dist + 1)

            if response <fastest and response != -1:
                fastest = response
        if fastest ==-1:
            return -1
        else:
            self.fastestPath[node] = fastest
            return fastest+1




    def CreateFastestPathHandle(self):
        start = [100,50,50]
        self.fastestPath =  np.zeros(len(self.index_to_coord), dtype=int) - 1
        start_index = self.coord_to_index[start[0],start[1],start[2]]
        self.currentPath = np.zeros(self.max_path, dtype=int) -1

        self.CreateFastestPath(start_index, 0)
        np.save(self.path + "path/" + str(self.frame) + "_path.npy", self.fastestPath)

