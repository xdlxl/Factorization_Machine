# Multi-thread-Factorization-Machine
1.A very efficient factorization machine c++ multi-thread implement，using google downpour method  
2.in the train file , each line contains a sample, in format
click \t impre \t feat_index1 \t val1 feat_index2 \t val2 ...
3.gcc4.8 surpport
4.support rank loss and logistic loss 
5.support read local .gz file or from hdfs 
6. config file feat_num & fm_index &　hdfs_conf must be provided
7. in 8-core machine,it can process 100 million samples (hundreds non zero features) one epoch in one hour
