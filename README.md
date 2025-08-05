# CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN

This repository accompanies the paper:

**CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN**  
Victor Kamel, Hanxueyu Yan, and Sean Chester (2025)

If you use this work, please cite the paper.

---

## 📬 Contact

For questions or collaboration inquiries, please reach out:

- Victor Kamel — [vkamel@cs.toronto.edu](mailto:vkamel@cs.toronto.edu)  
- Hanxueyu Yan — [hyan76131@uvic.ca](mailto:hyan76131@uvic.ca)  
- Sean Chester — [schester@uvic.ca](mailto:schester@uvic.ca)

---

## ⚙️ Build Instructions

### 1. On Compute Canada

First, load the necessary modules:

```bash
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cmake/3.27.7 cuda/12.2
```

Then build the project:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The executable `linear-scans` will be created.

To run:

```bash
./linear-scans
```

---

### 2. On Local Linux (e.g., CLOVER Workstation)

#### Requirements:
- CUDA 12.6  
- GCC 13.3  

#### Build Steps:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. [If using FAISS methods, add: -DLINK_FIASS=1]
make
```

#### To run:

```bash
./linear-scans 2
# or
./linear-scans 3
```

Supported arguments are listed in `/src/linear-scans.cu`, lines 31–42.  
Alternatively, you can view them by running:

```bash
./linear-scans
```

---

### 🧠 Supported Algorithms

```cu
enum class Algorithm
{
    bitonic,     // Data-parallel batch insertion inspired by bitonic merge sort
    warpwise,    // Warp-ballot-based insertion sort that adds one point at a time
    hubs,        // Spatio-graph-based with hubs and lower bounds
    hubs_ws,     // Same as 'hubs' but uses WarpSelect
    faiss,       // Linear-scan from Facebook/Meta (FAISS)
    faiss_ws,    // FAISS with WarpSelect
    faiss_bs,    // FAISS with BlockSelect
    treelogy_kdtree // GPU kNN search from Treelogy library
};
```

---

### 3. Included Baselines

Please refer to each baseline's original repository for build instructions.

#### 📝 Note for Arkade users:

Use the following command to run the modified Arkade version included in this repo:

```bash
./s01-knn filename nsearchpoints
```

Do **not** add the `radius_estimate` or `npoints` arguments as required by the original version.  
The radius selection is handled internally with a custom method that differs from the original release.

---

## 📊 Output Format Explanation

Example output:

```
4597372, (1000, 30, [6656711, 6572149, ])
```

This breaks down as:
- 4597372: Time (nanoseconds) for a warm-up unit test (cold cache avoidance)
- 1000: Dataset size
- 30: k-value used in kNN
- [6656711, 6572149]: Execution times (nanoseconds) for different kernels or runs

More examples:

```
4727741, (2000, 30, [6861110, 6684729])
4706599, (5000, 30, [7521143, 7475509])
4539267, (10000, 30, [8009394, 7982956])
4505187, (20000, 30, [9942419, 9912732])
4494838, (50000, 30, [15839078, 14783502])
4175195, (80000, 30, [20105605, 20517700])
4164274, (100000, 30, [24446526, 24451917])
4160988, (150000, 30, [29267678, 28326381])
3059219, (200000, 30, [36935429, 37059730])
3315912, (350000, 30, [67434637, 67339721])
3771468, (500000, 30, [105312322, 102915573])
```

