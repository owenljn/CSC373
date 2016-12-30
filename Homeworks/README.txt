Name: Jinnan Lu
Student number: 997698807
I worked alone

(1) The Algorithm
The algorithm contains two parts: generate random graph, solve MST problem. For the first part, the graph is saved as a 2D array, where (i,j) component represents the weight of edge that connect vertex i and j. The complexity for the first part is O(V^2), for large V, it can take a long time. Since the work to generate weights on different edges are independent, we can run them in parallel. For the second part, we use Prim algorithm to solve the MST problem: Divide the vertices into two groups, one that the connection in MST is already known, one that the connection is to be determined. We select vertex from the second group and add to the first gradually, one each time. The selected vertex has the property that it has the minimum distance to the first group among vertices in the second group. The vertices in the second group are stored in a BinaryHeap, ordered by their distance to the first group.

(2) Program Output
RandomMST
V		run time(in milliseconds)		output
10		0.200000						1.439313
100		1.800000						2.127473
1000	30.850000						2.884657
10000	3980.450000						3.848576

CircleMST
V		run time(in milliseconds)		output
10		0.250000						3.935144
100		1.950000						14.086866
1000	25.150000						44.061877
10000	4098.950000						141.039143

(3)
For RandomMST, the output grows like C*log(V), where C is a constant
For Circle, the output grows like C*V*log(V), where C is a constant
