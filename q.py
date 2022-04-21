class arrayQueue:
    def __init__(self):
        self.nodes = []
        self.keys = []
        self.size = 0

    def insert(self, edgetoadd, keyval):  #total complexity: O(1) + O(1) = O(1)
        self.nodes.append(edgetoadd)  # O(1)
        self.keys.append(keyval)  # O(1)
        self.size += 1

    def makequeue(self, nodelist, keylist):  # total complexity: O(n)
        self.nodes = list(nodelist)
        self.keys = list(keylist)
        self.size = len(keylist)

    def decrease_key(self, node, keyval): # total complexity: O(n)  -> I tried fixing this to be o(1) broke program
        # nindex = self.nodes.index(node)  # lookup, worst case is O(n)
        self.keys[node.node_id] = keyval       # O(1)

    def isempty(self):
        if self.size == 0:
            return True
        else:
            return False

    def maxdepth(self):
        maxdepth = 0
        for x in self.nodes:
            if maxdepth < x.getlevel():
                maxdepth = x.getlevel()
        return maxdepth

    # total complexity  O(n)
    def delete_min(self):  # Modified functionality to check node depth and prefer depth to absolute key value
        # add equivalent depths to a special list
        maxdepth = self.maxdepth()
        tieddepths = []
        depthkey = []
        for i in self.nodes:
            if i.getlevel() == maxdepth:
                tieddepths.append(i)
                depthkey.append(self.keys[self.nodes.index(i)])

        tiedmin = min(depthkey)
        index = self.keys.index(tiedmin)
        self.keys.remove(tiedmin)
        ret = self.nodes[index]
        self.nodes.remove(ret)
        self.size -= 1
        return ret



