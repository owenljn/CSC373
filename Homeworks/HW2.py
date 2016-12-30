import numpy as np
import sys

def randomMST(V):
    edges = np.random.uniform(0, 1, size=V)
    return edges
print (sys.argv[1:])
print (randomMST(sys.argv[1:]))
#if __name__ == "__main__":
#    randomMST(sys.argv[1:])