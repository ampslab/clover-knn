I. Treelogy: A Benchmark Suite for Tree Traversals
https://bitbucket.org/plcl/treelogy

Treelogy is a benchmark suite and an ontology for tree traversal kernels. 
Treelogy contains shared-memory (SHM), distributed-memory(DM), and GPU implementations of 9 tree applications
drawn from diverse domains. Further, some benchmarks are implemented with multiple tree types. 
Distributed-memory implementations of some of the benchmarks are part of the SPIRIT framework.

The applications and their different tree implementations are listed below:
 
1)Two-point correlation (with kd-, and vp-trees)
2)Nearest Neighbor (with kd-, vp-, and ball-trees)
3)K-Nearest Neighbor (with kd-, and ball-trees)
4)Barnes-Hut (with kd- and octrees)
5)Photon Mapping for Ray tracing (with kdtree)
6)Frequent itemset mining (with prefix tree)
7)Fast multipole method (with quad tree)
8)K-means clustering (with kdtree)
9)longest common prefix  (with suffix tree)

Treelogy benchmarks contain a baseline version (base) and an optimized baseline (block).  For those benchmarks where blocked execution (Jo et.al., OOPSLA11) is not possible (freqmine, kmeans, longestcommonprefix, fastmultipole), base represents the optimized baseline. GPU implementations of photon mapping, longestcommonprefix, freqmine are not yet available. 

Acknowledgements:
Y.Jo - harness, and the code for autotuning block sizes (Jo et.al. OOPSLA'11). 
J.Hbeika - GPU fastmultipolemethod code.
The baselines of the following benchmarks are adapted from other sources:
heatray - https://code.google.com/archive/p/heatray/. 
nearest neighbor with vantage point trees - http://stevehanov.ca/blog/index.php?id=130. 
barnes hut - Lonestar benchmark suite (http://iss.ices.utexas.edu/?p=projects/galois/lonestar).
freqmine - https://github.com/integeruser/FP-growth/
kmeans - https://github.com/vaivaswatha/kdkmeans-cuda
longestcommonprefix - http://www.geeksforgeeks.org/generalized-suffix-tree-1/ 

If you find this software useful in academic work, please cite the following publications:

1)N.Hegde, J. Liu, K. Sundararajah, and M. Kulkarni, "Treelogy: A Benchmark Suite for Tree Traversals", ISPASS'17. 
2)J. Liu, N. Hegde, and M. Kulkarni, "Hybrid cpu-gpu scheduling and execution of tree traversals", ICS'16.
3)Y. Jo and M. Kulkarni, "Enhancing locality for recursive traversals of recursive structures", OOPSLA'11.


II. Build instructions
1) Distributed-memory implementations require a working MPI installation and Boost Graph Library (BGL) to be installed.
If your working MPI installation is MPI_HOME, then first install BGL and build libraries that contain wrappers for lower level MPI functions.
Installing BGL:
  1)Download latest tar (boost_1_63_0.tar.bz2), extract "tar -xvf boost_1_63_0.tar.bz2". This should create a folder boost_1_63_0. Call the absolute path to this folder as BOOST_HOME.
  2)Goto boost_1_63_0 folder. Run "./bootstrap.sh --prefix=BOOST_HOME --with-libraries=mpi,graph_parallel,system" 
  3)Edit BOOST_HOME/project-config.jam. Add the following line at the end: "using mpi : MPI_HOME/bin/mpic++ ;" (exclude quotes and note that MPI_HOME/bin must contain mpic++ or mpiCC).
    Make sure $PATH contains the location of mpic++/mpiCC.
  4)Run "./b2". This creates a folder BOOST_HOME/stage. The folder must contain the newly built mpi, serialization, and graph_parallel libraries. 

Once BGL is installed, edit common/Makefile.common in treelogy/DM/spirit to update BOOST_HOME.
Run "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:BOOST_HOME/stage/lib".
Now you can go to any application folder located inside spirit and run make successfully.

III. Inputs
Only synthetic input generators (instructions to generate) are provided.

IV. Running
Look at scripts/AppExecution to get an idea of what parameters an application expects or how to run the programs.  

