This project focuses on parallelizing GCNs for recommendation systems. Since the goal of this project is the compare the speed of parallel and serial implementations we used a implementation of a GCN by Thomas Kipf as a serial baseline and made adjustments as needed for our parallelization. Links to all references and resources used can be found below. Other aspects of the GCN model have been modified for our needs but the changes we made are mainly in module.cu. The following is a brief summary our of results.

This was the time for training on 100 epochs.

Serial MM:
    pubmed: 13070.6 ms
    citeseer: 13629.2 ms
    cora: 4550.84 ms

Serial CSR MM:
    pubmed: 4046.37 ms
    citeseer: 658.447 ms
    cora: 465.893 ms

Parallel MM:
    pubmed: 
    citeseer: 
    cora: 

Parallel CSR MM:
    pubmed: 
    citeseer: 
    cora: 

References:

https://tkipf.github.io/graph-convolutional-networks/
