📄 Paper

This codebase accompanies the paper:

CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN
Victor Kamel, Hanxueyu Yan, and Sean Chester (2025)

If you use this work, please cite the paper.

📬 Contact

For questions or collaboration inquiries, please contact:

    Victor Kamel — vkamel@cs.toronto.edu

    Hanxueyu Yan — hyan76131@uvic.ca

    Sean Chester — schester@uvic.ca



========================================
         BUILD INSTRUCTIONS
========================================

-----------------------------
1. On Compute Canada
-----------------------------

First, load the necessary modules:

    module --force purge
    module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cmake/3.27.7 cuda/12.2

Then build using an out-of-source directory:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

This will create an executable called `linear-scans`.

Run the program as:

    ./linear-scans


-----------------------------
2. On Local Linux (e.g., Clover)
-----------------------------

Requirements:
    - CUDA 12.6
    - GCC 13.3

Build steps:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

Run the executable with arguments:

    ./linear-scans 2
    or
    ./linear-scans 3
    
    The arguments listing is:
    
    
-----------------------------
3. For Included Baselines
-----------------------------

Please refer to the original repositories of the baselines for their specific build instructions.

**Note for Arkade users:**

Use the following command to run the modified version included in this repo:

    ./s01-knn filename nsearchpoints

Do **not** include the `radius_estimate` and `npoints` argument as required in the original version.

The radius selection algorithm has been incorporated into the modified Arkade code in this repository.
It differs from the one used in the original release.

