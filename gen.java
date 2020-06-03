public class gen {

    public static void main(String[] args) {
        String[] nodes = {"10","20","30","40","50","60","70"};
        String[] edgeprobs = {"0.5","1"};
        String[] maxweights = {"20","40","60","80","100","120"};


        int jobid = 1;
        for(String edgeprob: edgeprobs){
            for (String node: nodes){
                for (String maxweight: maxweights) {
                    for(int i = 0; i < 5; i++){
                        String[] params = new String[4];
                        params[0] = node; // number of nodes
                        params[1] = edgeprob; // edge probability
                        params[2] = maxweight; // max weight of edges
                        params[3] = Integer.toString(jobid); // id
                        GraphGenerator.main(params);
                        System.out.println("Computed a graph " + jobid + " with " + node + " nodes, maxweight: " + maxweight + " and edge probability: " + edgeprob);
                        jobid++;
                    }
                }
            }
        }
    }
}
