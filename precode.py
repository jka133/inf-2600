import matplotlib.pyplot as plt
import numpy as np

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
        if hold_index != None:
            for i in range(index, hold_index, 1):
                new_configuration[i] = self.next_color(i, new_configuration)
        else: 
            for i in range(index, self.height, 1):
                new_configuration[i] = self.next_color(i, new_configuration)

        return CubeTower(new_configuration, self)
        

    def next_color(self, index, new_configuration):
        color = new_configuration[index]
        next_color_index = self.order.index(color) + 1 
        if (self.order.index(color) + 1) < len(self.order): 
            next_color_index = self.order.index(color) + 1 
        else:
            next_color_index = self.order.index(color) - 3

        return self.order[next_color_index]

# Example Usage
"""initial_configuration = ["red","blue","red","green"]
tower = CubeTower(initial_configuration)
tower.visualize()"""

def child_nodes(tower):
    lst = []
    for i in range(tower.height):
        for j in range(tower.height):
            if j < i:
                continue
            lst.append(tower.rotate_cube(i, j))
    
    return lst
# Implement the search algorithms here
def dfs_search(tower):

    stack = []
    for node in child_nodes(tower):
        print(node.configuration)
        
    
    

def bfs_search(tower):
    # Implement Breadth-First Search
    pass

def a_star_search(tower):
    # Implement A* Search
    pass

# Additional advanced search algorithm
# ...


if __name__ == '__main__':
    initial_configuration = ["yellow","red","blue","green"]
    tower = CubeTower(initial_configuration)
    #tower.visualize()
    #tower = dfs_search(tower)


    """tower = tower.rotate_cube(1, 2)
    tower = tower.rotate_cube(1, 2)
    tower = tower.rotate_cube(1, 2)
    tower = tower.rotate_cube(3,0)
    tower = tower.rotate_cube(3)"""

    #tower.visualize()
    """tower.check_cube()
    tower.get_path()"""
    dfs_search(tower)
    for ct in child_nodes(tower):
        continue
        ct.visualize_path()


    """tower2 = CubeTower(initial_configuration)
    dfs_search(tower2)
    """