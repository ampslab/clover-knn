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
