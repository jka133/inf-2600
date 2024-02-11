import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
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
        """
        Returns the next colour for the cube in the towers order
        """
        color = new_configuration[index]
        #print(self.order.index(color) + 1 )
        next_color_index = self.order.index(color) + 1 
        if (self.order.index(color) + 1) < len(self.order): 
            next_color_index = self.order.index(color) + 1 
        else:
            next_color_index = self.order.index(color) - 3

        return self.order[next_color_index]

def child_nodes(tower):
    """
    Returns a list of the possible tower to be made in one move from the tower given
    """
    lst = []
    for i in range(0, tower.height, 1):
        for j in range(i+1, tower.height, 1):
            lst.append(tower.rotate_cube(i, j))
    return lst

def unvisited(lst_to_check, lst_visited):
    """
    Returns a list of unvisited configurations, given the candidates and the visited ones
    """
    return [x for x in lst_to_check if x.configuration not in lst_visited]

# Implement the search algorithms here
@profile
def dfs_search(tower, stack = [], explored = [], depth = 0):
    """
    Depth first search. Visit the newest made configuration until it is solved.
    """
    stack.append(tower)

    while stack:
        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"DFS success after {depth} operations")
            #current.visualize_path()
            return current
        
        explored.append(current.configuration)
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored)

        stack = unvisited_child_nodes + stack
        
        depth += 1
@profile
def bfs_search(tower, stack = [], explored = [], depth = 0):
    """
    Breadth first search. Visit every child node from every tower until one is solved.
    """
    stack.append(tower)

    while stack:
        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"BFS success after {depth} operations")
            #current.visualize_path()
            return current
        
        explored.append(current.configuration)
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored)
        
        stack = stack + unvisited_child_nodes
        
        depth += 1

def heuristic(tower):
    """
    Returns the heuristic of a tower. The heuristic is the highest number of same-coloured blocks in the configuration
    Negative since it is a contrast to the steps taken to achive the configuration 
    """
    cols = []
    counts = []
    for x in tower.configuration:
        if x not in cols:
            cols.append(x)
            counts.append(tower.configuration.count(x))

    return - max(counts)

def steps(tower):
    """
    Returns the number of steps taken to achive the current config from the initial one
    """
    return len(tower.get_path())

def evaluation(tower):
    """
    Evaluates the steps and heuristic. Best case is lowest possible -> few steps and many same-coloured cubes
    """
    return steps(tower) + heuristic(tower)

@profile
def a_star_search(tower, stack = [], depth = 0):
    """
    Evaluates the tower with the best (lowest) evaluation to be explored next, until solved
    """
    stack.append(tower)

    while stack:
        current = stack.pop(0)
        if current.check_cube() == True:
            break

        depth += 1
        stack += child_nodes(current)
        stack.sort(key=evaluation) # Sorts list based on evaluation(tower) in ascending order
        
    print(f"A* success after {depth} operations")
    #current.visualize_path()
    return current

# Additional advanced search algorithm
# ...
@profile
def gfs_search(tower, depth = 0):
    """ Greedy first search. Evaluates the heuristic of the child nodes and uses the child with best heuristic each time """
    while tower.check_cube() != True:
        depth += 1
        children = child_nodes(tower)
        children.sort(key=heuristic)
        tower = children[0]
        
    print(f'GFS success after {depth} operations')
    #current.visualize_path()
    return tower

def search_result(search_method, tower):
    """
    Function to run the search algorithms with timing and vizualising them
    """
    print("\n" + "=" * 40 + "\n")  

    start = time.time()
    sollution = search_method(tower)
    time_taken = time.time() - start
    print(f"{search_method.__name__} search time: {time_taken:.6f} seconds")

    length = len(sollution.get_path())
    sollution.visualize_path()
    return length, time_taken

if __name__ == '__main__':

    initial1_configuration = ["green","red","blue","yellow","red","yellow"]#,"blue"]#,"green","blue","yellow"]#,"red","yellow"]
    tower1 = CubeTower(initial1_configuration)
    tower1.visualize()
    
    """initial2_configuration = ["green","red","blue","blue","red"]#,"yellow"]
    tower2 = CubeTower(initial2_configuration)
    tower2.visualize()
    
    initial3_configuration = ["green","red","blue","blue","red","yellow"]
    tower3 = CubeTower(initial3_configuration)
    tower3.visualize()"""
    
    search_result(a_star_search, tower1)
    search_result(gfs_search, tower1)
    #search_result(dfs_search, tower1)
    #search_result(bfs_search, tower1)