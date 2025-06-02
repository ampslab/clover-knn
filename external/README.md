# Build Instructions for FAISS

1. Set up modules on Compute Canada:
```
module --force purge;module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cmake/3.27.7 cuda/12.2 imkl/2023.2.0
```
2. Checkout the FAISS submodule:
```
git submodule update --init --recursive
```
3. Create build directory and set up makefiles from inside external/faiss:
```
cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_RAFT=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_C_API=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic -B build .
```
4. Build faiss from inside external/faiss (note: it is more friendly not to use *all* cores on the login node)
```
time make -C build -j4 faiss
```
