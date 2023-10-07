from cmath import inf
import numpy as np
import copy

from scipy.sparse import csr_matrix
import random



def randomSplit(Num):
    l = list(range(0, Num))
    random.shuffle(l)
    if (Num % 2 == 0):
        return [l[:Num//2], l[Num//2:]]
    else:
        return list((l[:Num//2+1], l[Num//2+1:]))


def spMatInit(str):
    f = open(str, "r")

    firstLine = f.readline().split(" ")
    netNum = int(firstLine[0])
    cellNum = int(firstLine[1])

    Nets = f.readlines()


    mat = np.zeros((cellNum, cellNum))


    for net in Nets:
        netCell = list(map(int, net.split(" ")[:-1]))
        
        netWeight = 1.0/(len(netCell)-1)

        for i in range(len(netCell)):
            for j in range(i+1, len(netCell)):
                mat[netCell[i]-1][netCell[j]-1] += netWeight
                mat[netCell[j]-1][netCell[i]-1] += netWeight

    print(mat) 
    
    SparseMat = csr_matrix(mat)

    SpMat = [ ([SparseMat[i].indices, SparseMat[i].data]) for i in range(cellNum)]

    splitNodes =  randomSplit(cellNum)
    remainNodes = copy.deepcopy(splitNodes)

    # print out all list in SpMat
    for i in range(cellNum):
        for j in range(len(SpMat[i][0])):
            print(SpMat[i][0][j], SpMat[i][1][j])
        
    exit()
    return SpMat, splitNodes, remainNodes




def swip(num1, num2):
    remainNodes[0].remove(num1)
    remainNodes[1].remove(num2)

    idx_1 = splitNodes[0].index(num1)
    idx_2 = splitNodes[1].index(num2)

    splitNodes[0][idx_1] = num2
    splitNodes[1][idx_2] = num1

def connections(a):
    
    E,I = 0,0
    node = SpMat[a][0]
    wieght = SpMat[a][1]

    for idx in range(len(node)):
        if (node[idx] in splitNodes[0]):
            I += wieght[idx]
            # print("I", node[idx], wieght[idx])
        else:
            E += wieght[idx]
            # print("E", node[idx], wieght[idx])

    return E-I

def nodeConnection(a,b):
    node = SpMat[a][0]
    wieght = SpMat[a][1]

    for idx, val in enumerate(node):
        if (val == b):
            return wieght[idx]

    return 0.0

def calCutSize(SpMat, splitNodes):
    E = 0

    for rightIdx in splitNodes[0]:
        node = SpMat[rightIdx][0]
        wieght = SpMat[rightIdx][1]

        for leftIdx in range(len(node)):
            if (node[leftIdx] in splitNodes[1]):
                E += wieght[leftIdx]
    return E

def KL(SpMat, splitNodes, remainNodes):
    cutSize = calCutSize(SpMat, splitNodes)
    MinCutSize = copy.deepcopy(cutSize)

    print(cutSize)

    count = 0
    print("\ninteration", count)
    print("CutSize:", cutSize)


    while (remainNodes[0] != [] and remainNodes[1] != []):
        max = -inf
        count += 1
        ll = []
        
        conn_1 = []
        conn_2 = []
        # save connections() for all remainNodes and save it into an array          
        for i in remainNodes[0]:
            conn_1.append(connections(i))
            
        for i in remainNodes[1]:
            conn_2.append(connections(i))
            
        
        for idx_1, val_1 in enumerate(remainNodes[0]): #for count, val_1 in values):
            for idx_2, val_2 in enumerate(remainNodes[1]):
                gain = conn_1[idx_1] - conn_2[idx_2] - 2*nodeConnection(val_1,val_2)
                if max < gain:
                    max = gain
                    ll = [val_1, val_2]
        print("\ninteration", count) 
        print("gain   :", max)
        cutSize = cutSize - max
        print("CutSize:", cutSize)
        if (cutSize < MinCutSize):
            MinCutSize = cutSize
        print("choose :", ll)
        swip(ll[0], ll[1])
        print("split  :", splitNodes)
        print("remain :", remainNodes, "\n")

    print("Best CutSize:", MinCutSize)



if __name__=="__main__":
    SpMat, splitNodes, remainNodes = spMatInit("fract.hgr")

    KL(SpMat, splitNodes, remainNodes)

