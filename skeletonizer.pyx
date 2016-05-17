import cv2
import numpy as np


class Skeletonizer(object):
    def __init__(self, I):
        self.binary_image = I.copy()
        self.height, self.width = self.binary_image.shape

    #check if the point tup is in the image
    def is_in_image(self, tup):
        return tup[0] >= 0 and tup[0] < self.height and tup[1] >= 0 and tup[1] < self.width

    def get_neighbour_points(self, i, j):
        #4-connectivity
        #point_list = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
        #8-connectivity
        point_list = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
        return filter(self.is_in_image, point_list)

    #template for north, south, east, west functions
    def is_template(self, image, i, j, i_n, j_n, n_white_neighbours):
        if n_white_neighbours > 1:
            if self.is_in_image((i_n, j_n)):
                if image[i_n, j_n] > 0:
                    return False
            return True
        else:
            return False

    def is_north(self, image, i, j, n_white_neighbours):
        return self.is_template(image, i, j, i - 1, j, n_white_neighbours)

    def is_south(self, image, i, j, n_white_neighbours):
        return self.is_template(image, i, j, i + 1, j, n_white_neighbours)

    def is_east(self, image, i, j, n_white_neighbours):
        return self.is_template(image, i, j, i, j + 1, n_white_neighbours)

    def is_west(self, image, i, j, n_white_neighbours):
        return self.is_template(image, i, j, i, j - 1, n_white_neighbours)

    def is_simple(self, image, i, j):
        #get 3x3 neighbourhood of the pixel (i, j)
        begin_x = max(0, i - 1)
        end_x = min(self.height, i + 2)
        begin_y = max(0, j - 1)
        end_y = min(self.width, j + 2)
        neighbourhood = image[begin_x:end_x, begin_y:end_y].copy().astype('u1')
        #count the number of connected components in neighbourhood
        n_components = cv2.connectedComponents(neighbourhood)[0] - 1
        #remove (i, j) from the hand
        image[i, j] = 0
        neighbourhood = image[begin_x:end_x, begin_y:end_y].copy().astype('u1')
        image[i, j] = 1
        n_components_new = cv2.connectedComponents(neighbourhood)[0] - 1
        #if the connectivity in the neighbourhood doesn't change, then the point is simple
        if n_components == n_components_new:
            return True
        else:
            return False

    def is_endpoint(self, image, i, j, n_white_neighbours):
        if n_white_neighbours == 1:
            return True
        else:
            return False

    def is_isolated(self, image, i, j, n_white_neighbours):
        if n_white_neighbours == 0:
            return True
        else:
            return False

    #we use this function to remove north, south, east and west points, depending on func parameter
    def check_points(self, image, func):
        changed = False
        res_image = image.copy()
        #find points, belonging to the had
        x_range, y_range = np.nonzero(image)
        for i, j in zip(x_range, y_range):
            #find the number of neighbours for the point
            n_white_neighbours = sum([image[n_i, n_j] for n_i, n_j in self.get_neighbour_points(i, j)])
            if func(image, i, j, n_white_neighbours):
                if self.is_simple(image, i, j):
                    if not (self.is_isolated(image, i, j, n_white_neighbours) or self.is_endpoint(image, i, j, n_white_neighbours)):
                        changed = True
                        res_image[i, j] = 0

        if changed:
            return res_image
        else:
            return None


    def find_skeleton(self):
        res_image = self.binary_image
        n_consecutive_fails = 0
        while True:
            #removing north
            new_image = self.check_points(res_image, self.is_north)
            if new_image is None:
                if n_consecutive_fails == 3:
                    #we have tried all types, time to finish
                    break
                else:
                    #we failed to remove any points here, but we might succeed later
                    n_consecutive_fails += 1
            else:
                res_image = new_image
                n_consecutive_fails = 0

            #removing south
            new_image = self.check_points(res_image, self.is_south)
            if new_image is None:
                if n_consecutive_fails == 3:
                    break
                else:
                    n_consecutive_fails += 1
            else:
                res_image = new_image
                n_consecutive_fails = 0

            #removing east
            new_image = self.check_points(res_image, self.is_east)
            if new_image is None:
                if n_consecutive_fails == 3:
                    break
                else:
                    n_consecutive_fails += 1
            else:
                res_image = new_image
                n_consecutive_fails = 0

            #removing west
            new_image = self.check_points(res_image, self.is_west)
            if new_image is None:
                if n_consecutive_fails == 3:
                    break
                else:
                    n_consecutive_fails += 1
            else:
                res_image = new_image
                n_consecutive_fails = 0


        return res_image


class SkelGraph(object):
    def __init__(self, skel_obj, I_skel):

        self.nodes = set()
        self.paths = dict()
        self.skel_obj = skel_obj
        self.I_skel = I_skel.copy()

        #finding the first endpoint
        x_range, y_range = np.nonzero(I_skel)
        first_endpoint = None
        first_direction = None
        for i,j in zip(x_range, y_range):
            neighbours = self.get_skel_neighbours(i, j)
            n_white_neighbours = len(neighbours)
            if n_white_neighbours == 1:
                first_endpoint = (i, j)
                first_direction = neighbours[0]
                break

        self.I_skel[first_endpoint[0], first_endpoint[1]] = 2
        self.nodes.add(first_endpoint)

        #computing the skeleton
        self.parse_skeleton(first_endpoint, first_direction)

        #populating some convinient data structures for later use

        self.connections = dict()

        self.lengths = dict()
        for points, path in self.paths.items():
            l = sum([np.linalg.norm((p1[0] - p2[0], p1[1] - p2[1])) for p1, p2 in zip(path[0:-1], path[1:])])
            self.lengths[points] = l

        self.degrees = dict()
        for p1, p2 in self.paths.keys():
            if p1 in self.degrees.keys():
                self.degrees[p1] += 1
                self.connections[p1].append(p2)
            else:
                self.degrees[p1] = 1
                self.connections[p1] = [p2]

            if p2 in self.degrees.keys():
                self.degrees[p2] += 1
                self.connections[p2].append(p1)
            else:
                self.degrees[p2] = 1
                self.connections[p2] = [p1]

    def get_path(self, node_from, node_to):
        if (node_from, node_to) in self.paths.keys():
            return self.paths[(node_from, node_to)]
        elif (node_to, node_from) in self.paths.keys():
            return self.paths[(node_to, node_from)][::-1]
        else:
            return None

    def get_skel_neighbours(self, i, j):
        neighbours = list(self.skel_obj.get_neighbour_points(i, j))
        return [(i1, j1) for i1, j1 in neighbours if self.I_skel[i1, j1] > 0]

    def parse_skeleton(self, point, direction):
        path = [point]
        #we move along the current path, until we reach the edge, or the point with more, than 2 neighbours
        while True:
            if self.I_skel[direction[0], direction[1]] > 1:
                # we have already been here
                return False
            else:
                self.I_skel[direction[0], direction[1]] = 2

            path.append(direction)
            neighbours = self.skel_obj.get_neighbour_points(direction[0], direction[1])
            neighbours = [(i, j) for i, j in neighbours if self.I_skel[i, j] > 0]
            neighbours.remove(point)
            n_neighbours = len(neighbours)
            if n_neighbours == 1:
                # continue moving
                point = direction
                direction = neighbours[0]
                continue
            elif n_neighbours == 0:
                # we've reached the edge, add it to the list
                self.nodes.add(path[-1])
                self.paths[(path[0], path[-1])] = path
                return True
            else:
                #explore other branches
                if direction not in self.nodes:
                    for new_direction in neighbours:
                        self.parse_skeleton(direction, new_direction)

                    self.nodes.add(direction)
                    self.paths[(path[0], path[-1])] = path
                return True
