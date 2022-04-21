class CostMatrix:

    def __init__(self, state, cost, matrix, src, dest, path):
        self.state = state
        self.cost = cost
        self.matrix = matrix
        self.path = path
        self.loc = (src, dest)
        self.size = len(path)

    def __eq__(self, other):
        if isinstance(other, CostMatrix):
            return self.state == other.state


    def __lt__(self, other):
        if isinstance(other, CostMatrix):
            if self.size == other.size:
                return self.cost < other.cost
            else:
                return self.size < other.size 

    def __gt__(self, other):
       if isinstance(other, CostMatrix):
            if self.size == other.size:
                return self.cost > other.cost
            else:
                return self.size > other.size 
    
    # digging deeper
    # def __lt__(self, other):
    #     if isinstance(other, CostMatrix):
    #         return  (self.cost // self.size) < (other.cost // other.size)
    
    # def __gt__(self, other):
    #     if isinstance(other, CostMatrix):
    #         return (self.cost // self.size) > (other.cost // other.size)

    def __hash__(self):
        return hash(self.getName())

    def getName(self):
        return str(self.state)

    def printMatrix(self):
        matrix = ""
        for elem in self.matrix:
            matrix += str(elem) + "\n"
        return matrix

    def __str__(self):
        return '(state={}\n loc={}\n cost={} \n path={}' \
               ' \n matrix= \n{} )'\
            .format(self.state, self.loc, self.cost,
                    self.path, self.matrix)

    def __repr__(self):
        return self.__str__()


class HeapPriorityQueue:

    def __init__(self):
        self.list = []
        self.size = 0
        self.map = {}
        self.max_size = 0

    # The isEmpty function checks if the queue is empty.
    def isEmpty(self):
        return len(self.list) == 0

    """
    Return the parent index of the node at the given index

    :param index: The index of the node to be removed
    :return: The index of the parent node.
    """

    def parent(self, index):
        return (index - 1) // 2

    """
        Given an index, return the index of the left child

        :param index: The index of the node to be retrieved
        :return: The index of the left child of the node at the given index.
    """

    def leftChild(self, index):
        return (2 * index) + 1

    """
    Return the index of the right child of the node at the given index

    :param index: The index of the node to be retrieved
    :return: The index of the right child of the node at the given index.
    """

    def rightChild(self, index):
        return ((2 * index) + 2)

    """
        If the current node is smaller than its parent, swap the nodes

        :param index: The index of the node to be bubbled up
    """

    def bubbleUp(self, index):
        self.size = len(self.list) - 1

        while index > 0 and self.list[self.parent(index)] > self.list[index]:
            # Swap parent and current node
            self.swap(self.parent(index), index)

            # Update index to parent of index
            index = self.parent(index)

    """
        If the current node is less than the left child, swap the current node with the left child. 

        If the current node is less than the right child, swap the current node with the right child. 

        If the current node is less than both the left and right child, do nothing.

        :param index: The index of the node to be bubbled down
    """

    def bubbleDown(self, index):

        min_val = index
        self.size = len(self.list) - 1

        # left child
        left = self.leftChild(index)

        # compare left child node
        if left <= self.size and self.list[left] < self.list[min_val]:
            min_val = left

        # right child
        right = self.rightChild(index)

        if right <= self.size and self.list[right] < self.list[min_val]:
            min_val = right

        if index != min_val:
            self.swap(index, min_val)
            self.bubbleDown(min_val)

    """
        Insert a node into the heap and then bubble it up to its proper position

        :complexity :  O(log n)

        :param node: The node to be inserted

    """

    def insert(self, node):

        self.list.append(node)
        self.size = len(self.list) - 1
        self.map[node] = self.size
        self.bubbleUp(self.size)
     

    """
        The first element in the list is replaced with the last element in the list. 

        The last element in the list is then bubbled down to its appropriate position. 

        The function returns the value of the element that was replaced.
         :complexity :  O(log n)
        :return: The minimum value in the heap.

    """

    def deleteMin(self):
        self.size = len(self.list) - 1
        min_val = self.list[0]
        del self.map[min_val]

        self.list[0] = self.list[self.size]
        self.map[self.list[0]] = 0

        del self.list[self.size]
       

        # update max PQ size 
        if self.max_size < self.size:
            self.max_size = self.size

        if len(self.list) != 0:
            self.bubbleDown(0)
        else:
            del self.map[min_val]

        return min_val


    """
        If the value of the element at the given index is less than the value of the element at the
        parent index, then swap the two elements

        :complexity :  O(log n)

        :param index: the index of the node to be bubbled up or down
        :param val: the value to be inserted into the heap
    """

    def decreaseKey(self, oldVal, val):
        index = self.map[oldVal]

        print("index of item ", index)

        # get the node from index
        old_val = self.list[index]

        # set the new value
        self.list[index] = val

        del self.map[oldVal]
        self.map[val] = index

        if val < old_val:
            self.bubbleUp(index)
        else:
            self.bubbleDown(index)



    def swap(self, i, j):
        
        # swap values
        temp = self.list[i]
        self.list[i] = self.list[j]
        self.list[j] = temp

        # update map index

        self.map[self.list[j]] = j
        self.map[self.list[i]] = i






