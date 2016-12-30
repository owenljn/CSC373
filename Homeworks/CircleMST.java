import java.util.*;
import java.util.concurrent.*;

public class CircleMST {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Please provide number of weights in command line.");
            System.exit(-1);
        }

        int V = Integer.valueOf(args[0]);

        if (V < 3) {
            System.out.println(String.format("V=%s is too small, at least 3.", V));
            System.exit(-2);
        }

        new CircleMST().simulate(V);
        //new CircleMST().statistics(V);
    }


    public void simulate(int V) {
        CircleRandomCompleteGraph graph = CircleRandomCompleteGraph.generate(V);
        double cost = SolveMST(graph);
        System.out.println(cost);
    }

    public void statistics(int V) {
        int simTimes = 20;
        int totalTime = 0;
        double totalCost = 0.0;
        for (int i = 0; i < simTimes; i++) {
            long start = System.currentTimeMillis();
            CircleRandomCompleteGraph graph = CircleRandomCompleteGraph.generate(V);
            totalCost += SolveMST(graph);
            long end = System.currentTimeMillis();
            totalTime += (end - start);
        }
        System.out.printf("%d\t%f\t%f\n", V, totalTime / (double)simTimes, totalCost / (double)simTimes);
    }

    public double SolveMST(CircleRandomCompleteGraph graph) {
        List<Node> nodes = generateNodes(graph.getV());
        Set<Node> remain = new HashSet<>(nodes);

        BinaryHeap Q = new BinaryHeap(nodes);
        double cost = 0.0;
        while (!Q.isEmpty()) {
            Node min = Q.extractMin();
            remain.remove(min);
            cost += min.key;

            for (Node n: remain) {
                double newWeight = graph.getWeight(n.id, min.id);
                if (newWeight < n.key) {
                    Q.decreaseKey(n.index, newWeight);
                    n.parent = min;
                }
            }
        }
        return cost;
    }

    private List<Node> generateNodes(int V) {
        List<Node> nodes = new ArrayList<>(V);
        for (int i = 0; i < V; i++)
            nodes.add(new Node(i));
        nodes.get(0).key = 0;
        return nodes;
    }

    static class BinaryHeap {
        int size = 0;
        private List<Node> array;

        public BinaryHeap(List<Node> array) {
            this.array = array;
            this.size = array.size();
        }

        public Node min() {
            return get(0);
        }

        public Node extractMin() {
            if (size < 1)
                throw new RuntimeException("Underflow");
            Node min = get(0);
            set(0, get(size - 1));
            size--;
            minHeapify(0);
            return min;
        }

        public boolean isEmpty() {
            return size == 0;
        }

        public int size() {
            return size;
        }

        public void buildHeap() {
            for (int i = size / 2 - 1; i >= 0; i--)
                minHeapify(i);
        }

        public void decreaseKey(int ind, double newKey) {
            get(ind).key = newKey;
            int p = parent(ind);
            while(ind > 0 && get(p).compareTo(get(ind)) > 0) {
                exchange(ind, p);
                ind = p;
            }
        }

        private void minHeapify(int i) {
            int l = left(i);
            int r = right(i);
            int minimum = 0;
            if (l < size && get(l).compareTo(get(i)) < 0)
                minimum = l;
            else
                minimum = i;
            if (r < size && get(r).compareTo(get(minimum)) < 0)
                minimum = r;
            if (minimum != i) {
                exchange(i, minimum);
                minHeapify(minimum);
            }
        }

        private void exchange(int i, int j) {
            Node tmp = get(i);
            set(i, get(j));
            set(j, tmp);
        }

        private void set(int i, Node n) {
            array.set(i, n);
            n.index = i;
        }

        private Node get(int i) {
            return array.get(i);
        }

        private int parent(int i) {
            return (i - 1) / 2;
        }

        private int left(int i) {
            return 2 * i + 1;
        }

        private int right(int i) {
            return 2 * i + 2;
        }
    }

    static class Node implements Comparable<Node> {
        private static final double MAX_WEIGHT = Double.MAX_VALUE;
        public int id;
        public int index;
        public double key = MAX_WEIGHT;
        public Node parent = null;

        public Node(int id) {
            this.id = id;
            this.index = id;
        }

        @Override
        public int compareTo(Node o) {
            return Double.compare(key, o.key);
        }
    }

    static class CircleRandomCompleteGraph {
        private final int V;
        private final double[][] weights;

        private static final int PARALLEL_GENERATE_THERSHOLD = 1000;

        private CircleRandomCompleteGraph(int V, double [][] weights) {
            this.V = V;
            this.weights = weights;
        }

        public int getV() {
            return V;
        }

        static class Point {
            public double x;
            public double y;
            public Point(double x, double y) {
                this.x = x;
                this.y = y;
            }
        }

        private static Point randomPointInUnixCircle(Random rand) {
            double alpha = 2.0 * Math.PI * rand.nextDouble();
            double r = rand.nextDouble() + rand.nextDouble();
            if (r > 1.0) r = 2.0 -r;
            return new Point(r * Math.cos(alpha), r * Math.sin(alpha));
        }

        private static List<Point> generatePoints(int V) {
            List<Point> points = new ArrayList<>(V);
            Random rand = new Random();
            for (int i = 0; i < V; i++)
                points.add(randomPointInUnixCircle(rand));
            return points;
        }

        private static double distance(Point a, Point b) {
            double deltaX = a.x - b.x;
            double deltaY = a.y - b.y;
            return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        }

        public static CircleRandomCompleteGraph generate(int V) {
            if (V < PARALLEL_GENERATE_THERSHOLD)
                return singleThreadGenerate(V);
            else
                return parallalGenerate(V);
        }

        private static CircleRandomCompleteGraph singleThreadGenerate(int V) {
            final List<Point> points = generatePoints(V);
            double [][] w = new double[V][V];
            Random rand = new Random();
            for (int row = 0; row < V; row++) {
                for (int col = 0; col < row; col++) {
                    w[row][col] = w[col][row] = distance(points.get(row), points.get(col));
                }
            }
            return new CircleRandomCompleteGraph(V, w);
        }

        //divide the work of calculating pairwise distance uniformly among the workers
        //the total units work is about 1 + 2 + ... + V, since we only calculate once for (i,j) and (j,i)
        //Since each row has different units of work to do, in order to divide the work uniformly among workers
        //we choose divide points: V/sqrt(n), sqrt(2) * V/sqrt(n), sqrt(2)^2 * V/sqrt(n), sqrt(2)^3 * V/sqrt(n), ...
        private static List<Integer> getDividePoints(int V, int n) {
            int ind = (int)(V / Math.sqrt(n));
            double SQRT2 = Math.sqrt(2.0);
            List<Integer> points = new LinkedList<>();
            points.add(0);
            while(ind < V - 1) {
                points.add(ind);
                ind = (int)(SQRT2 * ind);
            }
            points.add(V);
            return points;
        }

        private static CircleRandomCompleteGraph parallalGenerate(int V) {
            final List<Point> points = generatePoints(V);
            final double[][] w = new double[V][V];
            int n = (int)(Runtime.getRuntime().availableProcessors() * 0.7);
            ExecutorService executorService = new ThreadPoolExecutor(n, n, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingDeque<>());
            int step = V / n;
            List<Integer> divs = getDividePoints(V, n);
            int nThread = divs.size() - 1;

            final CountDownLatch latch = new CountDownLatch(nThread);
            for (int k = 0; k < nThread; k++) {
                final int start = divs.get(k);
                final int end = divs.get(k + 1) - 1;
                Runnable task = new Runnable() {
                    @Override
                    public void run() {
                        for (int i = start; i <= end; i++) {
                            for (int j = 0; j < i; j++) {
                                w[i][j] = w[j][i] = distance(points.get(i), points.get(j));
                            }
                        }
                        latch.countDown();
                    }
                };
                executorService.submit(task);
            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            executorService.shutdown();
            return new CircleRandomCompleteGraph(V, w);
        }

        double getWeight(int i , int j) {
            return weights[i][j];
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for(int row = 0; row < V; row++) {
                for(int col = 0; col < row; col++) {
                    sb.append(String.format("%f ", getWeight(row, col)));
                }
                sb.append("\n");
            }
            return sb.toString();
        }
    }
}
