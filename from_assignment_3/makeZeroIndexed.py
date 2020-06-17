# make zero indexed
import os

files = os.listdir()
print(files)
for fname in files:
    if (not "opt" in fname) and (not ".py" in fname) and (not "new" in fname):
        with open(fname, "r") as f:
            with open("new"+fname,"w") as fw:
                nodes = int(f.readline())
                fw.write("%d\n" % nodes)

                lines = f.readlines()
                for i, line in enumerate(lines):
                    edge = line.split()
                    weight = int(edge[2])
                    edge = edge[:2]
                    node_1, node_2 = int(edge[0]), int(edge[1])
                    fw.write("%d %d %d\n" % (node_1-1,node_2-1,weight))
            fw.close
        f.close