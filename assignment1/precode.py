import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
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
        path.append(current.configuration) # Added this as the original config should be included
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
        Returns the new configuration when cube "index" is rotated
        """
        color = new_configuration[index]

        if (self.order.index(color) + 1) < len(self.order): # If the next colour index is in range
            next_color_index = self.order.index(color) + 1 
        else: # Subtract the length of the colour order to arrive at the correct colour
            next_color_index = self.order.index(color) - len(self.order) + 1

        return self.order[next_color_index]

def child_nodes(tower):
    """
    Returns a list of the possible towers to be made in one move from the tower given
    """
    lst = []
    for i in range(0, tower.height, 1):
        for j in range(i+1, tower.height, 1):
            lst.append(tower.rotate_cube(i, j)) # Append the rotated tower objects
    return lst

def unvisited(lst_to_check, lst_visited):
    """
    Returns a list of unvisited configurations, given the candidates and the visited ones
    """
    return [x for x in lst_to_check if x.configuration not in lst_visited]

def dfs_search(tower, stack = [], explored = [], depth = 0):
    """
    Depth first search. Visit the newest made configuration until it is solved.
    """
    stack.append(tower) # List of towers to explore

    while stack:
        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"DFS success after {depth} operations")
            return current, depth
        
        explored.append(current.configuration) # Note the current as explored if not solved
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored) # Make unvisited child nodes

        stack = unvisited_child_nodes + stack # Add the new nodes first to the stack
        
        depth += 1

def bfs_search(tower, stack = [], explored = [], depth = 0):
    """
    Breadth first search. Visit every child node from every tower until one is solved.
    """
    stack.append(tower)
    # Very similar to DFS
    while stack:
        current = stack.pop(0)

        if current.check_cube() == True:
            print(f"BFS success after {depth} operations")
            return current, depth
        
        explored.append(current.configuration)
        
        children = child_nodes(current)
        unvisited_child_nodes = unvisited(children, explored)
        
        stack = stack + unvisited_child_nodes # Add the new nodes last in the stack as the breadth should be explored first
        
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

    return - max(counts) # Measure of the configurations distance to solution

def steps(tower):
    """
    Returns the number of steps taken to achive the current config from the initial one
    """
    return len(tower.get_path()) # The steps taken

def evaluation(tower):
    """
    Evaluates the steps and heuristic. Best case is lowest possible -> few steps and many same-coloured cubes
    """
    return steps(tower) + heuristic(tower) # The sum of moves and distance to solution

def a_star_search(tower):
    """
    Evaluates the tower with the best (lowest) evaluation to be explored next, until solved
    """
    stack = [tower]
    depth = 0

    while stack:
        current = stack.pop(0)
        if current.check_cube() == True:
            break

        depth += 1
        stack += child_nodes(current)
        # Sorts stack based on evaluation(tower) in ascending order
        stack.sort(key=evaluation) # The lowest evaluation is the least moves and best heuristic
        
    print(f"A* success after {depth} operations")
    return current, depth

def gfs_search(tower):
    """ Greedy first search. Evaluates the heuristic of the child nodes and uses the child with best heuristic each time """
    depth = 0
    while tower.check_cube() != True:
        depth += 1
        children = child_nodes(tower)
        children.sort(key=heuristic) # Only evaluate the heuristic of the current nodes child nodes
        tower = children[0] # Pick the best next step, without regard of previous towers
        
    print(f'GFS success after {depth} operations')
    return tower, depth

def search_result(search_method, tower):
    """
    Function to run the search algorithms with timing and memory tracing and vizualising them
    """

    tracemalloc.start() # Start memory trace
    start = time.time() # Start time capture
    solution, n_operations = search_method(tower) # Perform algorithm on tower
    peak_memory = tracemalloc.get_traced_memory()[1]
    time_taken = time.time() - start
    tracemalloc.stop()

    length = len(solution.get_path())
    solution.visualize_path()
    return length, n_operations, time_taken, peak_memory

def bar_plot(algorithm_names, solution_len_lst, n_operations_lst, time_taken_lst, memory_used_lst):
    """
    Function to plot solution length, memory used, time taken and moves made for the algorithms
    """
    # Set up the figure and axes
    # Bar plot for solution length
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(algorithm_names, solution_len_lst, color='skyblue')
    ax1.set_title('Solution Length by Algorithm')
    ax1.set_ylabel('Solution Length')
    ax1.set_xlabel('Algorithm')
    plt.show()

    # Bar plot for number of operations
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(algorithm_names, n_operations_lst, color='blueviolet')
    ax2.set_title('Operations Made by Algorithm')
    ax2.set_ylabel('Operations Made')
    ax2.set_xlabel('Algorithm')
    plt.show()

    # Bar plot for time taken
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(algorithm_names, time_taken_lst, color='lightgreen')
    ax3.set_title('Time Taken by Algorithm')
    ax3.set_ylabel('Time Taken (seconds)')
    ax3.set_xlabel('Algorithm')
    plt.show()

    # Bar plot for memory used
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.bar(algorithm_names, memory_used_lst, color='salmon')
    ax4.set_title('Memory Used by Algorithm')
    ax4.set_ylabel('Memory Used (MB)')
    ax4.set_xlabel('Algorithm')
    plt.show()


if __name__ == '__main__':
    # Tower configurations
    config1 = ["green","red","blue"]
    config2 = ["green","red","blue","yellow"]
    config3 = ["green","red","blue","yellow","red"]
    config4 = ["blue","red","blue","red","yellow"]
    config5 = ["green","red","blue","yellow","red","green"]
    # Tower initializations
    tower1,tower2,tower3,tower4,tower5 = CubeTower(config1),CubeTower(config2),CubeTower(config3),CubeTower(config4),CubeTower(config5)
    towers = [tower1, tower2, tower3, tower4, tower5]

    algorithms = [a_star_search, gfs_search, dfs_search, bfs_search]
    algorithm_names = ['A*', 'Greedy', 'DFS', 'BFS']

    # Solving each tower and reviewing them
    for t in towers:
        solution_len_lst, n_operations_lst, time_taken_lst, memory_used_lst = [],[],[],[]
        for algo in algorithms:
            solution_len, n_operations, time_taken, memory_used = search_result(algo, t)
            solution_len_lst.append(solution_len)
            n_operations_lst.append(n_operations)
            time_taken_lst.append(time_taken)
            memory_used_lst.append(memory_used)
        bar_plot(algorithm_names, solution_len_lst, n_operations_lst, time_taken_lst, memory_used_lst)