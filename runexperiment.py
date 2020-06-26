import maxcut
import Ant

import numpy as np
import ACO
import matplotlib.pyplot as plt
import runexperiment as rn

import multiprocessing
import copy

# run experiment
if __name__ == "__main__":
    # rn.doGridsearchExperiment()
    rn.doGlobalTest()

def doGlobalTest():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    instances_directory = 'testinstances/'
    opt_directory = 'opts/'
    instancenamelist = ["maxcut_random_16_0.4_100_instance_0","maxcut_random_32_0.4_100_instance_1","maxcut_random_64_0.4_100_instance_2","maxcut_random_128_0.4_100_instance_3","maxcut_random_256_0.4_100_instance_4","maxcut_random_512_0.4_100_instance_5"]
    # instancenamelist = ["maxcut_random_16_0.4_100_instance_0","maxcut_random_32_0.4_100_instance_1"]
    plt.ion()

    f = open("ACOend_results.txt","a")
    f.write("filename \t\t\t\t\t\t\t evaluations \t mean \t std \t maxfitness \n")
    # f2 = open("ACOrun_results.txt","w")
    for instancename in instancenamelist:
        fig,ax = plt.subplots()

        instance = maxcut.MaxCut(instancename+".txt", instances_directory, opt_directory)
        numAnts = 20
        max_evals = 10000000000
        max_gens = 200

        # best parameters for instance newL50_1_opt2078 without local search, rho = 0.2
        rho = 0.3 # result for with local search on N=128, 0.3 or 0.35
        ph_max=100
        ph_min=0.1
        alpha = 0.7

        ACO_instance = ACO.ACO_BBO2(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha)

        X = 10
        x, y, e, maxfit = rn.executeXtimes(X,ACO_instance)
        ax.errorbar(x, y, e, linestyle='None', marker="x",label=instancename)
        ax.legend(loc="lower right")
        plt.draw(), plt.pause(1e-4)

        f.write("%s \t %.0f \t %.2f \t %.5f \t %.0f \n" % (instancename,x[-1],y[-1],e[-1],maxfit))
        f.flush()
        # f.write("end mean: %.2f \n" % y[-1])
        # f.write("end std: %.5f \n" % e[-1])
        # f.write("best fitness found: %.0f" % maxfit)

    plt.ioff()
    f.close()
    plt.show()
    


def doGridsearchExperiment():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    instances_directory = 'instances/'
    instances_directory = 'testinstances/'
    # instances_directory = 'from_assignment_3/'
    opt_directory = 'opts/'
    # instancename = "maxcut_4x4_1_1_donut"
    # instancename = "maxcut_2x2_1_1_donut"
    instancename = 'newL25_2' #opt = 530
    # instancename = 'newL12_1' # opt = 127
    # instancename = 'newL12_2' # opt = 124
    instancename = "newL50_1_opt2078"
    # instancename = "newL50_2_opt2056"
    instancename = "maxcut_random_128_0.4_100_instance_3"

    instance = maxcut.MaxCut(instancename+".txt", instances_directory, opt_directory)
    numAnts = 20
    max_evals = 1000000
    max_gens = 200

    X = 10

    # For GBO:
    # rho_list = np.arange(0.01,0.51,0.1)
    # alpha_list = np.arange(0.75,2,0.25)
    # alpha_list = np.arange(0.5,1.5,0.1)

    # For BBO:
    # rho_list = np.arange(0.01,0.51,0.1)
    # alpha_list = np.arange(0.6,1.1,0.1)

    # For BBO2, in neighbourhood of:
    # rho = 0.2
    # ph_max=100
    # ph_min=0.1
    # alpha = 0.7

    rho_list = np.arange(0.15,0.4,0.05)
    alpha_list = np.arange(0.6,1,0.1)

    beta_list = [1]
    ph_min_list = [0.1]
    ph_max_list = [100]
    rn.gridSearch(X,instance,numAnts,max_evals,max_gens,rho_list,alpha_list,beta_list,ph_min_list,ph_max_list,BBO=True)

    plt.show()

def executeXtimes(X,ACO_instance):
    """
    Takes an ant colony optimization instance and executes/averages it over x>=1 times
    """
    # ACO_instance.run()

    # avg_archiveElitistList = np.zeros((X,np.shape(ACO_instance.archiveElitistList)[0]))
    # avg_archiveElitistList[0] = ACO_instance.archiveElitistList
    # evallist = ACO_instance.numEvalsList

    # for i in range(1,X):
    #     # ACO_instance.initAgain()
    #     # ACO_instance.run()
    #     # avg_archiveElitistList[i] = ACO_instance.archiveElitistList

    p = []
    manager = multiprocessing.Manager()
    avg_archiveElitistList = manager.dict()
    evallist = manager.dict()
    for i in range(0,X):
        ACO_instancenew = copy.deepcopy(ACO_instance)
        p.append(multiprocessing.Process(target=minifunction, args=(ACO_instancenew, avg_archiveElitistList,evallist,i)))
        p[-1].start()

    for j in p:
        j.join()
        
    return evallist.values()[0], np.mean(avg_archiveElitistList.values(),axis=0), np.std(avg_archiveElitistList.values(),axis=0), np.max(avg_archiveElitistList.values())

def minifunction(ACO_instance,avg_archiveElitistList, evallist, i):
    ACO_instance.initAgain()
    ACO_instance.run()
    avg_archiveElitistList[i] = ACO_instance.archiveElitistList
    evallist[i] = ACO_instance.numEvalsList

def gridSearch(X,instance,numAnts,max_evals,max_gens,rho_list,alpha_list,beta_list,ph_min_list,ph_max_list,BBO=True):
    plt.ion()
    fig,ax = plt.subplots()
    ax.set_xlabel("# evaluations")
    ax.set_ylabel("fitness")
    lab = ""

    # cm = plt.get_cmap('gist_rainbow')
    cm = plt.get_cmap('tab20')
    markers = ["x","^"]
    markerind = 0
    NUM_COLORS  = len(rho_list)*len(alpha_list)*len(beta_list)
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])


    for rho in rho_list:
        for alpha in alpha_list:
            for beta in beta_list:
                for ph_min in ph_min_list:
                    for ph_max in ph_max_list:
                        if BBO:
                            ACO_instance = ACO.ACO_BBO2(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha)
                            lab = r"$\rho = %.2f, \alpha = %.2f $" % (rho,alpha)
                        else:
                            ACO_instance = ACO.ACO_GBO(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,beta)
                            lab = r"$\rho = %.2f, \alpha = %.2f, \beta = %.2f $" % (rho,alpha,beta)

                        x, y, e,_ = rn.executeXtimes(X,ACO_instance)
                        ax.errorbar(x, y, e, linestyle='None', marker=markers[markerind],label=lab)
                        ax.legend(loc="lower right")
                        plt.draw(), plt.pause(1e-4)
                        markerind = 1 - markerind

    plt.ioff()