#pragma once

#include "spatial.cuh"

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <cassert>

namespace { // anonymous
} // namespace anonymous

namespace faiss_brute_force {

template <class R>
void knn_gpu(std::size_t n, R *data, std::size_t q, R *queries, std::size_t k, idx_t *results, R *distances)
{
    faiss::gpu::StandardGpuResources res;
    res.setDefaultNullStreamAllDevices();
    res.setTempMemory(0);
    faiss::gpu::GpuDistanceParams args;
    args.metric             = faiss::MetricType::METRIC_L2;
    args.metricArg          = 0;
    args.k                  = k;
    args.dims               = dim;
    args.vectors            = data;
    args.vectorsRowMajor    = true;
    args.numVectors         = n;
    args.queries            = queries;
    args.queriesRowMajor    = true;
    args.numQueries         = q;
    args.outDistances       = distances;
    args.ignoreOutDistances = true;
    args.outIndicesType     = faiss::gpu::IndicesDataType::I32;
    args.outIndices         = results;
    args.device             = -1;

    faiss::gpu::bfKnn(&res, args);
}

} // namespace faiss_brute_force
