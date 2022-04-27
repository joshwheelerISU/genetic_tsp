#!/usr/bin/python3

from random import shuffle
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
# 	from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
from TSP import *
from q import arrayQueue
from copy import deepcopy


class snode:
    def __init__(self):
        path = []
        matrix = []
        cost = 0
        nodeid = 0

    def getlevel(self):
        count = 0
        for x in self.path:
            count = count + 1
        return count


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        print(self.initialize_population(10))
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        bssf = None
        tspCalculator = TSPCalculator(ncities)
        start_time = time.time()

        adj = np.zeros((tspCalculator.N, tspCalculator.N))
        for i in range(ncities):
            for j in range(ncities):
                adj[i][j] = cities[i].costTo(cities[j])

        while not foundTour and time.time() - start_time < time_allowance:
            cost_matrix = tspCalculator.greedy_solve(adj)

            route = [cities[i] for i in cost_matrix.path[:-1]]
            bssf = TSPSolution(route)
            bssf.cost = cost_matrix.cost

            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = tspCalculator.bffs_updates
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    def branchAndBound(self, time_allowance=60.0):
        # var init - O(n)
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        pruned = 0
        bssf = None
        start_time = time.time()

        # blank n*n matrix to use later (O(n^2))
        nnblank = []
        for x in range(ncities):
            ydown = []
            for y in range(ncities):
                ydown.append(0)
            nnblank.append((ydown))

        # priority queue
        pq = arrayQueue()

        # algorithm body
        # start initial cost matrix for node 0
        rcm = nnblank
        for x in range(ncities):
            for y in range(ncities):
                val = cities[x].costTo(cities[y])
                rcm[x][y] = val

        # reduce that rcm for the first time to represent the rcm of the root node
        lim, rcm = self.reduceMatrix(rcm)

        # drop that root node in
        rn = snode()
        rn.cost = lim
        rn.matrix = rcm
        rn.path = [0]
        rn.nodeid = 0
        pq.insert(rn, rn.cost)

        # lets get a quick and dirty BSSF - Using the above random tour method.
        defaulttour = self.defaultRandomTour()
        bssf = defaulttour["soln"]
        newbsf = None
        boundtest = bssf.cost
        pathtest = bssf.route
        pmaxsize = 0
        totalnodescreated = 1

        # start the main algorithm loop
        while time.time() - start_time < time_allowance and pq.isempty() != True:  # Worst Case, 60 seconds.
            # update maximum pq count
            if pmaxsize < pq.size:
                pmaxsize = pq.size

            # pop a node off the queue
            curnode = pq.delete_min()
            matrix = curnode.matrix

            # check to see if we should just drop this node immediately
            if (curnode.cost < bssf.cost):
                min = -1
                mincost = curnode.cost
                if len(curnode.path) == ncities:
                    # we have a complete path, check to make sure we can go back to the beginning
                    if matrix[curnode.nodeid][0] != float('inf'):
                        curnode.cost = curnode.cost + matrix[curnode.nodeid][0]
                        nroute = []
                        for i in curnode.path:
                            nroute.append(cities[i])
                        bssf = TSPSolution(nroute)
                        count = count + 1
                        foundTour = True
                else:
                    for it in range(ncities):  # O(n)
                        if it not in curnode.path:  # make sure that we haven't been there before
                            if matrix[curnode.nodeid][it] + curnode.cost < boundtest and matrix[curnode.nodeid][
                                it] != float('inf'):  # trim if the solution is worse than the initial BSSF
                                min = it
                                travelcost = matrix[curnode.nodeid][it]

                                # make a new node to push onto the queue
                                pnode = snode()
                                pnode.matrix = [row[:] for row in matrix]
                                pnode.path = deepcopy(curnode.path)
                                pnode.path.append(it)
                                pnode.nodeid = it

                                # set rows and columns to infinity
                                for it in range(ncities):
                                    pnode.matrix[curnode.nodeid][it] = float('inf')
                                    pnode.matrix[it][pnode.nodeid] = float('inf')

                                # reduce the matrix
                                addcost, pmatrix = self.reduceMatrix(pnode.matrix)

                                # push the new node onto the priority queue
                                pnode.matrix = pmatrix
                                pnode.cost = curnode.cost + addcost + travelcost
                                pq.insert(pnode, pnode.cost)
                                # update node count
                                totalnodescreated = totalnodescreated + 1
                            else:
                                # we're pruning this path
                                pruned = pruned + 1
            else:
                pruned = pruned + 1

        # return setup - O(n)
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = pmaxsize
        results['total'] = totalnodescreated
        results['pruned'] = pruned + pq.size
        return results

    def reduceMatrix(self, rcm):
        # min of each col
        cities = self._scenario.getCities()
        ncities = len(cities)
        reduce = []
        for x in range(ncities):
            min = float('inf')
            for y in range(ncities):
                if rcm[x][y] <= min:
                    min = rcm[x][y]
            # go back through and reduce that col
            if (min != float('inf')):
                reduce.append(min)
                for y in range(ncities):
                    rcm[x][y] = rcm[x][y] - min
        # now go back and do the same in the opposite direction
        for y in range(ncities):
            min = float('inf')
            for x in range(ncities):
                if rcm[x][y] <= min:
                    min = rcm[x][y]
            # go back through and reduce that col
            if (min != float('inf')):
                reduce.append(min)
                for x in range(ncities):
                    rcm[x][y] = rcm[x][y] - min
        tot = 0
        for x in reduce:
            tot = tot + x
        return tot, rcm

    # <summary>
    # This is the entry point for the algorithm you'll write for your group project.
    # </summary>
    # <returns>results dictionary for GUI that contains three ints: cost of best solution,
    # time spent to find best solution, total number of solutions found during search, the
    # best solution found.  You may use the other three field however you like.
    # algorithm</returns>
    # 1. call greedy algorithm function to get bssf
    # 2. call init population function
    # 3. call function to retrieve 3 random samples from population, each with a
    # different weight depending on their cost
    # 4. call crossover function
    # 5. determine if mutation will occur, if so, call mutation function
    # 6. compare the result to the current best results and switch paths accordingly
    def fancy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        # finding the initial bssf - shouldn't count towards time
        bssf = self.greedy()['soln']

        start_time = time.time()

        # initialize the population with a size of (50)
        population = self.initialize_population(50)
        population[0] = bssf

        # main loop body definition
        while time.time() - start_time < time_allowance:
            # mate selection
            first, second, third = self.get_random_paths(population)
            m = self.weight_and_select(first, second, third)
            # breed
            new_path = self.crossover(bssf.route, m.route)
            child = TSPSolution(new_path)
            if m.cost > child.cost:
                # toss the parent, replace with child
                population[population.index(m)] = child
                # additional check for if this is the new bssf
                if bssf.cost > child.cost:
                    bssf = child
                    count = count + 1
        # final return preparation
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def weight_and_select(self, f, s, t):
        sumsol = f.cost + s.cost + t.cost
        if(sumsol < float('inf')):
            prob1 = f.cost / sumsol * 100
            prob2 = s.cost / sumsol * 100
            prob3 = t.cost / sumsol * 100
            dieroll = random.randint(0, 100)

            if dieroll > prob3:
                return t
            elif dieroll > prob2:
                return s
            else:
                return f
        else:
            return t

    def initialize_population(self, num_of_generations):
        cities = self._scenario.getCities()

        population = []

        for i in range(num_of_generations):
            list_copy = cities[1:]
            shuffle(list_copy)
            cities[1:] = list_copy
            new_cities = deepcopy(cities)
            population.append(TSPSolution(new_cities))

        return population

    # simple function used for sorting list of TSPSolutions
    def get_TSP_cost(self, path):
        return path.cost

    # function takes in the population pool and selects 3 random unique solutions
    # they are then sorted in a list and returned in ascending order of cost
    def get_random_paths(self, population):
        # 3 variables, can be adjusted to accommodate returning a larger
        # amount of samples from the population
        a, b, c = 0, 0, 0
        # select random samples from the population pool.
        while a == b == c:
            a = random.randint(0, len(population) - 1)
            b = random.randint(0, len(population) - 1)
            c = random.randint(0, len(population) - 1)
            # we now have 3 unique indexes from the population pool, so return them
        return population[a], population[b], population[c]

    def weighted_selection(self):
        pass

    def get_mutation(self, givenpath):
        cities = self._scenario.getCities()
        mutationvalid = False
        a = 0
        b = 0
        while a == b:
            a = random.randint(1, len(givenpath) - 1)
            b = random.randint(1, len(givenpath) - 1)
            # while (a == b):
            #     # try again, until we get random mutation points that aren't the same
            #     a = random.randint(1, len(givenpath) - 1)
            #     b = random.randint(1, len(givenpath) - 1)
            # # check to see if the cities are interchangeable
            # if givenpath[a - 1].costTo(givenpath[b] != float('inf')) and givenpath[b].costTo(
            #         givenpath[a + 1] != float('inf')) and givenpath[b - 1].costTo(givenpath[a] != float('inf')) and \
            #         givenpath[a].costTo(givenpath[b + 1] != float('inf')):
            #     mutationvalid = True
        # we've found a valid mutation, carry out the swap
        save = givenpath[a]
        givenpath[a] = givenpath[b]
        givenpath[b] = save
        return givenpath

    # function
    def crossover(self, bssf, rand_path):
        # take half of the best stored path
        # floor value in case of odd length
        new_path = bssf[:math.floor(len(bssf) / 2)]

        # Now take half of the rand_path and combine 
        # the 2 to create the new path
        for i in rand_path:
            if i not in new_path:
                new_path.append(i)
        if random.randint(0,100) > 80:
            new_path = self.get_mutation(new_path)
        return new_path

