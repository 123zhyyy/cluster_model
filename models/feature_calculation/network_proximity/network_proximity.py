
import sys
import os
import pandas as pd
sys.path.append(".../linjin/")
import toolbox
from toolbox import wrappers
file_name=".../gene_ppi.txt"

filename1=open(r'.../Cluster.txt')
filename2=open(r'.../drug_target.txt')

list=[]
for j in filename1:
        L=j.strip("\r\n")
        L2=filter(None,L.split("\t"))
        disease_targets=L2[1:]
print(disease_targets)
for i in filename2:
    #try:
        L=i.strip("\r\n")
        L2=filter(None,L.split("\t"))
        drug_targets=L2[1:]
        print(drug_targets)
    ##################network proximity calculation
        print(drug_targets)
        #print(disease_targets)
        network = wrappers.get_network(file_name, only_lcc = True)
        nodes_from =drug_targets
        nodes_to =disease_targets
        #distance=" closest"
        #"closest", "shortest", "kernel", "center", "jorg-closest"
        d,z,(mean,sd) = wrappers.calculate_proximity(network, nodes_from, nodes_to, min_bin_size = 2, seed=452456, distance="closest")
        print (d, z, (mean, sd)) 
      #(1-3)/4

        list.append([L2[0], d, z, (mean, sd)])
        list.append("\n")
        df=pd.DataFrame(list,columns=["d", "z","mean","sd"])
        df.to_csv(".../results.csv",index=False)
 
    #except Exception:
        #pass
