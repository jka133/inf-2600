import matplotlib.pyplot as plt
import numpy as np
import time

class CubeTower:
    def __init__(self, configuration, parent=None):
        """
        Initializes the cube tower with a given configuration.
        :param configuration: A list of the front-facing colors of the cubes in the tower, starting from the bottom.
        :param parent: The parent node of the current node. (can be used for tracing back the path)
        """
        self.order = ['red', 'blue', 'green','yellow']
        self.configuration = configuration
        self.height = len(configuration)
        self.parent = parent

    def visualize(self):
        """
        Visualizes the current state of the cube tower showing only the front-facing side.
        """
        fig, ax = plt.subplots()
        cube_size = 1  # Size of the cube

        for i, cube in enumerate(self.configuration):
            # Draw only the front-facing side of the cube
            color = cube
            rect = plt.Rectangle((0.5 - cube_size / 2, i), cube_size, cube_size, color=color)
            ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()

    def visualize_path(self):
        """
        Visualizes the path taken to reach this state from the initial state.
        """
        path = self.get_path()
        fig, ax = plt.subplots()
        cube_size = 1

        for i, configuration in enumerate(path):
            for j, cube in enumerate(configuration):
                color = cube
                rect = plt.Rectangle((i + 0.5 - cube_size / 2, j), cube_size, cube_size, color=color)
                ax.add_patch(rect)

        ax.set_xlim(0, len(path))
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()

    def get_path(self):
        """
        Retrieves the path taken to reach this state from the initial state.
        """
        path = []
        current = self
        while current.parent is not None:
            path.append(current.configuration)
            current = current.parent
        path.append(current.configuration)
        path.reverse()
        return path
    
    def check_cube(self):
        """
        Check if the cube tower is solved, i.e. all cubes are of the same color.
        """
        return len(set(self.configuration)) == 1

    def rotate_cube(self, index, hold_index=None):
        """
        Rotates a cube and all cubes above it, or up to a held cube.
        :param index: The index of the cube to rotate.
        :param hold_index: The index of the cube to hold, if any.
        """
        # Implement the rotation logic
        new_configuration = [x for x in self.configuration]
        if hold_index == 0:
            hold_index = None
        
        
        if hold_index != None:
            for i in range(index, hold_index, 1):
                new_configuration[i] = self.next_color(i, new_configuration)
        else: 
            for i in range(index, self.height, 1):
                new_configuration[i] = self.next_color(i, new_configuration)

        return CubeTower(new_configuration, self)    

    def next_color(self, index, new_configuration):
        color = new_configuration[index]
        #print(self.order.index(color) + 1 )
        next_color_index = self.order.index(color) + 1 
        if (self.order.index(color) + 1) < len(self.order): 
            next_color_index = self.order.index(color) + 1 
        else:
            next_color_index = self.order.index(color) - 3

        return self.order[next_color_index]

def child_nodes(tower):
    lst = []
    for i in range(0, tower.height, 1):
        for j in range(0, tower.height, 1):
            if j <= i:
                continue
            lst.append(tower.rotate_cube(i, j))
    return lst

def unvisited(lst_to_check, lst_visited):
    unvisited_lst = []
    for elem in lst_to_check:
        if (elem.configuration in lst_visited):
            continue
        unvisited_lst.append(elem)
    return unvisited_lst

# Implement the search algorithms here
def dfs_search(tower, stack = [], explored = [], depth = 0):

    stack.append(tower)

    while stack:

        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"DFS success after {depth} operations")
            #current.visualize_path()
            return current
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored)

        stack = unvisited_child_nodes + stack
        
        depth += 1
        explored.append(current.configuration)

def bfs_search(tower, stack = [], explored = [], depth = 0):
    # Implement Breadth-First Search

    stack.append(tower)

    while stack:


        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"BFS success after {depth} operations")
            #current.visualize_path()
            return current
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored)

        stack = stack + unvisited_child_nodes
        
        depth += 1
        explored.append(current.configuration)

def heuristic(tower):

    cols = []
    counts = []
    for x in tower.configuration:
        if x not in cols:
            cols.append(x)
            counts.append(tower.configuration.count(x))

    return - max(counts)

def a_star_search(tower, depth = 0):
    # Implement A* Search
    """ There is not step measure since there is only allowed one operation, only heuristic is calculadet"""

    while tower.check_cube() != True:
        depth += 1
        children = child_nodes(tower)
        current = children[0]
        for child in children[1:]:
            if heuristic(child) < heuristic(current):
                current = child
        tower = current
        
    print(f"A* success after {depth} operations")
    #current.visualize_path()
    return current

# Additional advanced search algorithm
# ...
def best_indexes(config):
    indx, hold_index = 0, 0
    cols = []
    counts = []
    for x in config:
        if x not in cols:
            cols.append(x)
            counts.append(config.count(x))

    best_colour = cols[counts.index(max(counts))]
    config = [1 if col == best_colour else 0 for col in config]

    if config[0] == True:
        for i in range(len(config)):
            if config[i] != True:
                indx = i
                break

        for j in range(indx, len(config)):
            if config[j] == True:
                hold_index = j
                break
            if j == len(config) - 1:
                hold_index = 0

    else:
        indx = 0
        for j in range(len(config)):
            if config[j] == True:
                hold_index = j
                break
            if j == len(config) - 1:
                hold_index = 0

    return indx, hold_index

def self_defined_search(tower, depth = 0):
    """ Greedy first search?"""
    while tower.check_cube() != True:
        depth += 1
        indx, hold_index = best_indexes(tower.configuration)
        tower = tower.rotate_cube(indx, hold_index)

    print(f'Self defined success after {depth} operations')
    #tower.visualize_path()
    return tower

def search_result(search_method, tower):
    print("\n" + "=" * 40 + "\n")  

    start = time.time()
    sollution = search_method(tower)
    end = time.time()
    print(f"{search_method.__name__} search time: {time.time() - end:.6f} seconds")
    sollution.visualize_path()

if __name__ == '__main__':
    initial_configuration = ["red","yellow","green","red"]
    tower = CubeTower(initial_configuration)
    tower.visualize()
    
    search_result(a_star_search, tower)
    search_result(self_defined_search, tower)
    search_result(dfs_search, tower)
    search_result(bfs_search, tower)