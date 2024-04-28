This project focuses on parallelizing GCNs for recommendation systems. Since the goal of this project is the compare the speed of parallel and serial implementations we used a implementation of a GCN by Thomas Kipf as a serial baseline and made adjustments as needed for our parallelization. Links to all references and resources used can be found in the project report. The parallelization techniques are in module.cu. Below are instructions on how to run the project and  a brief summary our of results for training on 100 epochs.

This project was run on CARC with the following commands (call the first two once):  
$ module load nvidia-hpc-sdk  
$ module load gcc/8.3.0  
$ make  
$ ./exec/gcn-seq <name of data>  

Serial MM:  
pubmed: 13070.6 ms  
citeseer: 13629.2 ms  
cora: 4550.84 ms  

Serial CSR MM:  
pubmed: 4046.37 ms  
citeseer: 658.447 ms  
cora: 465.893 ms  

Parallel MM:  
pubmed: 2425.62 ms  
citeseer: 377.448 ms  
cora: 292.327 ms  

Parallel CSR MM:  
pubmed: 2309.39 ms  
citeseer: 360.57 ms  
cora: 282.32 ms  

Full Parallel Implementation:  
pubmed: 1332.05 ms  
citeseer: 220.166 ms  
cora: 133.126 ms  
