### Build Instructions (Compute Canada)

First load the correct modules:
```
module --force purge;module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cmake/3.27.7 cuda/12.2
```

Then using an out-of-source build directory, build with cmake:
```
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This should create an executable called `linear-scans` that you can run without arguments.

### Profiling Instructions (Compute Canada)

Use the _NSight Compute command line_ utility to get basic profile information:

```
ncu ./linear-scans
```

For more detailed statistics, add extra group arguments:

```
ncu --section SpeedOfLight --section LaunchStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats ./linear-scans
```

To get event-level details, e.g., for warp stats:

```
ncu --metrics group:smsp__pcsamp_warp_stall_reasons ./linear-scans
```

For more details, see [the Nvidia documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
