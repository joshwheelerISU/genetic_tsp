import random

from Tools import *
import copy
import math
import time


class TSPCalculator:

    def __init__(self, n):
        self.stateNo = 0
        self.priorityQueue = HeapPriorityQueue()
        self.N = n
        self.bffs_cost_matrix = None
        self.bffs = math.inf
        self.pruned = 0
        self.bffs_updates = 0
        self.solFound = 0

    def reset(self):
        self.stateNo = 0
        self.pruned = 0
        self.bffs_updates = 0
        self.solFound = 0

    def initialize(self, matrix):
      # Initializing the problem.

        self.greedy_solve(matrix)
        self.reset()
        # create source cost matrix
        cost_matrix = CostMatrix(self.get_state(), 0, matrix, 0 , 0, [0])

        # reduce matrix and update cost
        # Time and Space - O(n^2)
        cost_matrix.cost = self.reduce_matrix(cost_matrix)

        # add to priority queue
        # Time - O(log n)
        self.priorityQueue.insert(cost_matrix)
    
    def greedy_solve(self, matrix):
        """
        For each node, find the cheapest path to the next node, and if there is a path back to the
        source, update the best path
        
        :param matrix: a 2D array of integers representing the cost of traveling between each pair of
        nodes
        :return: The cost matrix with the best solution
        """
        start = time.time()
        self.reset()

        cost = 0
        path = []

        # Space - O(n)
        # Complexity time - O(n^3)

        for length in range(self.N):
            i = random.randint(1, self.N) -1
            path.append(i)
            min_val = math.inf
            min_loc = -1

            # Time - O(n)
            while len(path) <= self.N:

                # Time - O(n)
                for j in range(len(matrix[i])):
                    if j != i and j not in path:

                        if min_val > matrix[i][j]:
                            min_val = matrix[i][j]
                            min_loc = j

                # check if a path exists
                if min_loc != -1:
                    cost += min_val
                    path.append(min_loc)
                    i = min_loc
                    min_val = math.inf
                    min_loc = -1
                else:
                    # There is no current path
                    break

            # all nodes are visited
            if len(path) == self.N:
                # check if there is a path back to source
                if  matrix[path[len(path) -1]][path[0]] != math.inf:
                    cost+=  matrix[path[len(path) -1]][path[0]]
                    path.append(path[0])

                    self.bffs = cost
                    self.bffs_cost_matrix = CostMatrix(-1, self.bffs, matrix, path[len(path) - 2], path[0], path)

                    end = time.time()
                    print("Time ", end - start)
                    return self.bffs_cost_matrix

                    # # update bffs
                    # if self.bffs > cost:
                    #     self.bffs = cost
                    #     self.bffs_cost_matrix = CostMatrix(-1, self.bffs, matrix, path[len(path) -2], path[0], path)
                    #     self.update_bbfs()
                    #
                    # # del path
                    # path = []
                    # cost = 0

                # no path to source
                else:
                    # del path
                    path = []
                    cost = 0
            # there is no path in this route
            else:
                # del path
                path = []
                cost = 0

        end = time.time()
        print("Time ", end - start)


        return self.bffs_cost_matrix

    def branch_bound_solve(self, matrix, time_allowance):
        """
        The function takes a matrix as input and returns the best path and cost
        
        :param matrix: the cost matrix
        :return: The best solution found so far.
        """
        start_time = time.time()
        # initalize problem
        self.initialize(matrix)

        # Worst-case Complexity: O(n^2 * 2^n)
        # runs until queue is empty and within time range
        while not self.priorityQueue.isEmpty() and time.time() - start_time < time_allowance:

            # get min cost matrix
            node = self.priorityQueue.deleteMin()

            # if lower bound is less than
            # current bffs expand
            if node.cost < self.bffs:

                # current node index
                i = node.loc[1]

                # expand possible paths
                # O(n - 1)
                for j in range(self.N):

                    # if i and j are equal
                    if i == j:
                        continue

                    # if node has been visited
                    if j in node.path:
                        continue

                    # create next node in search state
                    next_matrix = copy.deepcopy(node.matrix)
                    new_path = node.path.copy()
                    new_path.append(j)

                    # child state
                    next_node = CostMatrix(self.get_state(), node.cost,
                                           next_matrix, i , j , new_path)

                    # reduce matrix
                    # Time & Space - O(n^2)
                    self.make_infinity(i, j, next_node)

                    # reached all nodes
                    if len(next_node.path) == self.N:

                        # complete tour
                        if next_node.matrix[next_node.loc[1]][0] == 0:
                            next_node.path.append(0)
                            self.update_solutions_found()

                            # if cost is better, update bffs
                            if next_node.cost < self.bffs:
                                self.bffs = next_node.cost
                                self.bffs_cost_matrix = next_node
                                self.update_bbfs()

                    # if cost is better, add to queue
                    elif next_node.cost < self.bffs:
                        self.priorityQueue.insert(next_node)
                    else:
                        self.prune_state()
            else:
                self.prune_state()


        end_time = time.time()
        print("Time Taken: ", end_time - start_time)
        return self.bffs_cost_matrix

    def reduce_matrix(self, cost_matrix):
        """
        It takes a matrix and returns the lower bound of the matrix
        
        :param cost_matrix: The cost matrix of the problem
        :return: The bound of the matrix
        """

        row_min = []
        column_min = []
        bound = 0
        reduced_matrix = cost_matrix.matrix

        # Get row minimum
        # Time - O(n^2)
        # Space - O(n)
        for elem in reduced_matrix:
            row_min.append(min(elem))


        # Subtract the row minimum
        # Time - O(n^2)
        # Space - O(n^2)
        for i, min_val in enumerate(row_min):
            for j, elem in enumerate(reduced_matrix[i]):
                if min_val == math.inf:
                    continue
                reduced_matrix[i][j] = elem - min_val

        # Get column minimum
        # Time - O(n^2)
        # Space - O(n)
        for j in range(len(reduced_matrix[0])):
            col_min = math.inf
            for i in range(len(reduced_matrix)):
                if col_min > reduced_matrix[i][j]:
                    col_min = reduced_matrix[i][j]
            column_min.append(col_min)

        #print("min cols ", column_min, "\n")

        # Subtract the column minimum
        # Time - O(n^2)
        # Space - O(n^2)
        for j in range(len(reduced_matrix[0])):
            for i in range(len(reduced_matrix)):
                if column_min[j] == math.inf:
                    continue
                reduced_matrix[i][j] = reduced_matrix[i][j] - column_min[j]

        # add row residual
        # O(n^2)
        for i in range(len(row_min)):
            if row_min[i] == math.inf and i  in cost_matrix.path[:-1]:
                bound = bound + 0
            else:
                bound = bound + row_min[i]

        # add column residual
        #  O(n^2)
        for j in range(len(column_min)):
            if column_min[j] == math.inf and j  in cost_matrix.path[1:]:
                bound = bound + 0
            else:
                bound = bound + column_min[j]

        #print("bound of matrix: ", bound)

        return bound

    def make_infinity(self, row, column, cost_matrix):
        """
        It takes a matrix, makes the row and column of the current position infinity, and then reduces
        the matrix
        
        :param row: the row of the matrix that we're currently on
        :param column: the column of the matrix that we're currently on
        :param cost_matrix: This is the matrix that we're working with
        """
        # copy previous matrix
        temp_matrix = cost_matrix.matrix

        # get position score from matrix
        # get matrix cost

        # #print(temp_matrix)
        pos_cost = cost_matrix.matrix[row][column] + cost_matrix.cost

        # make row infinity
        # time - O(n)
        # space - O(n^2)
        for i in range(len(temp_matrix[row - 1])):
            temp_matrix[row][i] = math.inf

        # make column infinity
        # time - O(n)
        # space - O(n^2)
        for i in range(len(temp_matrix)):
            temp_matrix[i][column] = math.inf

        # make other paths infinity
        temp_matrix[column][row] = math.inf

        # check if matrix is reduced
        if not self.check_reduce_matrix(temp_matrix):
            # Space & Time -  O(n^2)
            cost_matrix.cost = self.reduce_matrix(cost_matrix) + pos_cost
        else:
            cost_matrix.cost  = pos_cost
     
    def check_reduce_matrix(self, matrix):
        """
        The function checks if the rows and columns of the matrix are reduced
        
        :param matrix: the matrix to be reduced
        :return: a boolean value.
        """

        # array to hold minimum values
        zeros = []

        # check if rows are reduced
        # O(n^2)
        for row in matrix:
            zeros.append(min(row))


        # check if columns are reduced
        # O(n^2)
        for j in range(len(matrix[0])):
            col_min = math.inf
            for i in range(len(matrix)):
                if col_min > matrix[i][j]:
                    col_min = matrix[i][j]
            zeros.append(col_min)

    

        # if element not equal to 0 and infinity
        # return false
        for elem in zeros:
            if elem != 0 and elem != math.inf:
                return False

        return True

    def get_state(self):
        """
        It returns the current state number and increments the state number by 1
        :return: The stateNo is being returned.
        """
        self.stateNo = self.stateNo + 1
        return self.stateNo

    def prune_state(self):
        """
        It adds 1 to the pruned variable.
        """
        self.pruned = self.pruned + 1

    def update_bbfs(self):
        """
        The function updates the number of times the breadth first search algorithm has been run
        """
        self.bffs_updates += 1

    def update_solutions_found(self):
        """
        It takes a list of numbers and returns the sum of the numbers.
        """
        self.solFound += 1

