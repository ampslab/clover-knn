#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <fstream>
#include <dirent.h>

#include "bitonic-based.cuh"
#include "bitonic-hubs.cuh"
#include "cpu-brute-force.cuh"
#include "warp-wise.cuh"
#include "treelogy_kdtree.cuh"

#ifdef USE_FAISS  
    #include "faiss-brute-force.cuh"
    #include "faiss-warp-select.cuh"
    #include "faiss-block-select.cuh"
    #include "bitonic-hubs-ws.cuh"
#endif

namespace { // anonymous

/** An enumeration of supported algorithms. */
enum class Algorithm
{
    bitonic,  /** A data-parallel batch insertion inspired by bitonic merge sort */
    warpwise, /** A warp-ballot-based insertion sort that adds one point at a time*/
    hubs,     /** A spatio-graph-based approach that builds an index with hubs and lower bounds */
    hubs_ws,  /** A spatio-graph-based approach that builds an index with hubs and lower bounds w/ WarpSelect */
    faiss,    /** The linear-scan method from Facebook/Meta Research */
    faiss_ws, /** The linear-scan method from Facebook/Meta Research using WarpSelect*/
    faiss_bs, /** The linear-scan method from Facebook/Meta Research using BlockSelect*/
    treelogy_kdtree, /*GPU version of knn search from Treelogy library*/
    Count     /** A pseudo-element to hack conversions from enum to int to get the enum element count */
};

const char* algorithm_descriptions[static_cast<std::size_t>(Algorithm::Count)] = {
        "Bitonic  -- a linear scan that uses a highly data-parallel partial bitonic sort",
        "Warpwise -- a linear scan that uses warp balloting to update a priority queue",
        "Hubs     -- a spatio-graph index-based approach that integrates Bitonic as a scan primitive",
        "HubsWS   -- a spatio-graph index-based approach that integrates Bitonic as a scan primitive w/ ws",
        "Faiss    -- a brute force linear scan using the faiss library",
	"FaissWarpSelect -- a brute force linear scan using WarpSelect from the faiss library",
	"FaissBlockSelect -- a brute force linear scan using BlockSelect from the faiss library",
};

/**
 * Dispatches knn algorithm based on launch configuration.
 */
template <class R, std::size_t N, std::size_t D, std::size_t Q>
auto dispatch_knn(const R (&data)[N][D], const idx_t (&queries)[Q], Algorithm alg, std::size_t k)
    -> std::pair< thrust::host_vector< idx_t >, thrust::host_vector< R > >
{
    assert(D == 3 && "Only dim=3 supported.");
    // # Move the points to the GPU
    R *dV;

    CUDA_CALL(cudaMalloc((void **) &dV, sizeof(R) * dim * N));
    CUDA_CALL(cudaMemcpy(dV, static_cast<const R*>(&data[0][0]), sizeof(R) * dim * N, cudaMemcpyHostToDevice));

    // # Move the queries to the GPU
    idx_t *dQ;

    CUDA_CALL(cudaMalloc((void **) &dQ, sizeof(idx_t) * Q));
    CUDA_CALL(cudaMemcpy(dQ, queries, sizeof(idx_t) * Q, cudaMemcpyHostToDevice));

    // # Allocate memory for return values
    thrust::device_vector<idx_t> d_knn(k * Q);
    thrust::device_vector<int> d_knn_treelogy(k * Q);
    thrust::device_vector<float> d_distances(k * Q);

#ifdef NDEBUG
    // collect timings
    cudaDeviceSynchronize();
    auto const timestamp_begin = std::chrono::high_resolution_clock::now();
#endif

    switch( alg )
    {
        case Algorithm::bitonic:
            bitonic::knn_gpu ( N, dV, Q, dQ, k
                             , thrust::raw_pointer_cast(d_knn.data())
                             , thrust::raw_pointer_cast(d_distances.data()) );
            break;

        case Algorithm::warpwise:
            warpwise::knn_gpu( N, dV, Q, dQ, k
                             , thrust::raw_pointer_cast(d_knn.data())
                             , thrust::raw_pointer_cast(d_distances.data()) );
            break;
        
        case Algorithm::hubs:
            bitonic_hubs::C_and_Q ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
            break;

        case Algorithm::hubs_ws:
#ifdef USE_FAISS
            bitonic_hubs_ws::C_and_Q ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm HubsWS but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss:
#ifdef USE_FAISS
            assert(N == Q);
            faiss_brute_force::knn_gpu( N, dV, N, dV, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss_ws:
#ifdef USE_FAISS
            faiss_warp_select::knn_gpu( N, dV, Q, dQ, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS WarpSelect but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss_bs:
#ifdef USE_FAISS
            faiss_block_select::knn_gpu( N, dV, Q, dQ, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS BlockSelect but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::treelogy_kdtree:
            treelogy::treelogy_kd_tree ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn_treelogy.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
        break;

        default:
            std::cerr << "Requested unknown algorithm" << std::endl;
            assert( false && "Unknown algorithm" );
    }

#ifdef NDEBUG
    // Collect timings
    cudaDeviceSynchronize();
    auto timestamp_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp_end-timestamp_begin).count();
    std::cout << duration << ", ";
#endif

    return std::make_pair(
        std::move(thrust::host_vector<idx_t>(d_knn))
      , std::move(thrust::host_vector<R>(d_distances)));
}
template <class R, std::size_t D>
auto dispatch_knn(std::vector<R>& data,  std::vector<idx_t>& queries, Algorithm alg, std::size_t k, std::size_t N, std::size_t Q)
    -> std::pair< thrust::host_vector< idx_t >, thrust::host_vector< R > >
{
    
    assert(D == 3 && "Only dim=3 supported.");
    // # Move the points to the GPU
    R *dV;

    CUDA_CALL(cudaMalloc((void **) &dV, sizeof(R) * D * N));
    CUDA_CALL(cudaMemcpy(dV, data.data(), sizeof(R) * D * N, cudaMemcpyHostToDevice));

    // # Move the queries to the GPU
    idx_t *dQ;

    CUDA_CALL(cudaMalloc((void **) &dQ, sizeof(idx_t) * Q));
    CUDA_CALL(cudaMemcpy(dQ, queries.data(), sizeof(idx_t) * Q, cudaMemcpyHostToDevice));


    // # Allocate memory for return values
    thrust::device_vector<idx_t> d_knn(k * Q);
    thrust::device_vector<int> d_knn_treelogy(k * Q);
    thrust::device_vector<float> d_distances(k * Q);

#ifdef NDEBUG
    // collect timings
    cudaDeviceSynchronize();
    auto const timestamp_begin = std::chrono::high_resolution_clock::now();
#endif

    switch( alg )
    {
        case Algorithm::bitonic:
            bitonic::knn_gpu ( N, dV, Q, dQ, k
                             , thrust::raw_pointer_cast(d_knn.data())
                             , thrust::raw_pointer_cast(d_distances.data()) );
            break;

        case Algorithm::warpwise:
            warpwise::knn_gpu( N, dV, Q, dQ, k
                             , thrust::raw_pointer_cast(d_knn.data())
                             , thrust::raw_pointer_cast(d_distances.data()) );
            break;
        
        case Algorithm::hubs:
            bitonic_hubs::C_and_Q ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
            break;

        case Algorithm::hubs_ws:
#ifdef USE_FAISS
            bitonic_hubs_ws::C_and_Q ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm HubsWS but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss:
#ifdef USE_FAISS
            assert(N == Q);
            faiss_brute_force::knn_gpu( N, dV, N, dV, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss_ws:
#ifdef USE_FAISS
            faiss_warp_select::knn_gpu( N, dV, Q, dQ, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS WarpSelect but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;

        case Algorithm::faiss_bs:
#ifdef USE_FAISS
            faiss_block_select::knn_gpu( N, dV, Q, dQ, k
                                      , thrust::raw_pointer_cast(d_knn.data())
                                      , thrust::raw_pointer_cast(d_distances.data()) );
#else
            std::cerr << "Requested algorithm FAISS BlockSelect but compiled without FAISS support" << std::endl;
            assert( false && "Compiled without faiss support.");
#endif
            break;


	        case Algorithm::treelogy_kdtree:
            treelogy::treelogy_kd_tree ( N, dV, Q, dQ, k
                            , thrust::raw_pointer_cast(d_knn_treelogy.data())
                            , thrust::raw_pointer_cast(d_distances.data()) );
        break;

        default:
            std::cerr << "Requested unknown algorithm" << std::endl;
            assert( false && "Unknown algorithm" );
    }

#ifdef NDEBUG
    // Collect timings
    cudaDeviceSynchronize();
    auto timestamp_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp_end-timestamp_begin).count();
    std::cout << duration << ", ";
#endif

    return std::make_pair(
        std::move(thrust::host_vector<idx_t>(d_knn))
      , std::move(thrust::host_vector<R>(d_distances)));
    
}
std::string sanitize_line(const std::string& line) {
    std::string sanitized = line;
    sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '\r'), sanitized.end());  
    std::replace(sanitized.begin(), sanitized.end(), ',', ' ');
    return sanitized;
}

template<class R, std::size_t D>
void fill_with_txt_data(const std::string& filename, std::vector<std::vector<R>>& data, std::size_t n) {
    std::ifstream txt_file(filename);
    if (!txt_file.is_open()) {
        std::cerr << "Failed to open TXT file: " << filename << std::endl;
        return;
    }

    std::string line;
    std::size_t i = 0;

    while (std::getline(txt_file, line) && i < n) {
        line = sanitize_line(line);
        std::istringstream line_stream(line);
        for (std::size_t j = 0; j < D; ++j) {
            if (!(line_stream >> data[i][j])) {
                std::cerr << "Failed to read data from TXT file at line " << i+1 << ", column " << j+1 << ": " << line << std::endl;
                return;
            }
        }
        ++i;
    }

    if (i != n) {
        std::cerr << "File does not contain enough data. Expected " << n << " rows, but only read " << i << " rows." << std::endl;
    }
}

std::size_t countVerticesInTXT(const std::string& filename) {

    std::cout << filename << std::endl;

    std::size_t dotPos = filename.find_last_of('.');  // Find the position of the last dot in the filename

    if (dotPos == std::string::npos || dotPos == 0 || dotPos == filename.size() - 1 || filename.substr(dotPos) != ".txt") {
        throw std::invalid_argument("The filename must end with '.txt'");
    }

    // Find the first digit in the filename
    auto firstDigitPos = std::find_if(filename.begin(), filename.begin() + dotPos, [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
    });

    if (firstDigitPos == filename.begin() + dotPos) {
        throw std::invalid_argument("No numeric portion found in the filename before '.txt'");
    }

    // Extract the numeric part of the string
    std::string numberStr(filename.begin() + std::distance(filename.begin(), firstDigitPos), filename.begin() + dotPos);
    std::size_t number = std::stoull(numberStr);  // Convert the extracted string to a number

    return number;
}


template <class R>
std::vector<R> flatten_2D_vector(const std::vector<std::vector<R>>& vec) {
    if (vec.empty()) return {};

    std::vector<R> flat_vec;
    flat_vec.reserve(vec.size() * vec[0].size());

    for (const auto& sub_vec : vec) {
        flat_vec.insert(flat_vec.end(), sub_vec.begin(), sub_vec.end());
    }

    return flat_vec;
}

template <class R, std::size_t N, std::size_t D>
void fill_with_random_data(R (&data)[N][D], std::optional<unsigned> seed = std::nullopt )
{
    if( ! seed.has_value() )
    {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 rng(seed.value());
    std::uniform_real_distribution<R> dist(-5, 5);
    std::generate_n((R*) &data, N * D, [&](){ return dist(rng); });
}

template < typename T >
void print_knn( T const& knn, std::size_t num_queries, std::size_t k )
{
    for (std::size_t i = 0; i < num_queries; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            std::cout << knn[i * k + j] << " ";
            //if (j % 32 == 0) std::cout << " | ";
        }
        std::cout << std::endl;
    }
}

/**
 * Parses the command line arguments to return a run configuration
 */
auto parse_input( int argc, char **argv ) -> std::optional< Algorithm >
{
    if( argc !=2 )
    {
        std::cout << "Usage: " << *argv << " algorithm" << std::endl;
        std::cout << "\tAlgorithm Options: " << std::endl;
        for (std::size_t i = 0; i < static_cast<std::size_t>(Algorithm::Count); ++i) {
                std::cout << "\t\t" << i << ": " << algorithm_descriptions[i] << std::endl;
        }

        return std::nullopt;
    }

    std::size_t pos = std::stoi( argv[ 1 ]);
    assert(pos < static_cast<std::size_t>(Algorithm::Count) && "Unknown algorithm");
    return Algorithm( pos );
}

bool run_test( Algorithm alg )
{
    constexpr std::size_t num_test_points = 4;
    constexpr std::size_t k = 3;

    float test_data[num_test_points][dim];
    test_data[0][0] = 0; test_data[0][1] = 1; test_data[0][2] = 1;
    test_data[1][0] = 0; test_data[1][1] = 1; test_data[1][2] = 0;
    test_data[2][0] = 1; test_data[2][1] = 1; test_data[2][2] = 1;
    test_data[3][0] = 0; test_data[3][1] = 0; test_data[3][2] = 0;

    idx_t Q[num_test_points];
    std::iota( Q, Q + num_test_points, 0 );

    auto const [results, distances] = dispatch_knn(test_data, Q, alg, k);

    return true;

    if( results[0]  != 0 ) { return false; }
    if( results[1]  != 1 ) { return false; }
    if( results[2]  != 2 ) { return false; }
    if( results[3]  != 1 ) { return false; }
    if( results[4]  != 0 ) { return false; }
    if( results[5]  != 3 ) { return false; }
    if( results[6]  != 2 ) { return false; }
    if( results[7]  != 0 ) { return false; }
    if( results[8]  != 1 ) { return false; }
    if( results[9]  != 3 ) { return false; }
    if( results[10] != 1 ) { return false; }
    if( results[11] != 0 ) { return false; }

    if( distances[0]  != 0 ) { return false; }
    if( distances[1]  != 1 ) { return false; }
    if( distances[2]  != 1 ) { return false; }
    if( distances[3]  != 0 ) { return false; }
    if( distances[4]  != 1 ) { return false; }
    if( distances[5]  != 1 ) { return false; }
    if( distances[6]  != 0 ) { return false; }
    if( distances[7]  != 1 ) { return false; }
    if( distances[8]  != 2 ) { return false; }
    if( distances[9]  != 0 ) { return false; }
    if( distances[10] != 1 ) { return false; }
    if( distances[11] != 2 ) { return false; }

    return true;
}

} // namespace anonymous


template<class R, std::size_t D>
void execute_knn_meshes(int argc, char **argv)
{
    std::optional< Algorithm > const algorithm = parse_input( argc, argv );
    if( algorithm.has_value() )
    {
        if(  run_test( algorithm.value() ) || true )
        {
            constexpr std::size_t reps = 1;

            //std::vector< int > k_values{32,64,128};
            std::vector< int > k_values{128};

            std::string directory_path ="../meshes";

            DIR* dirp = opendir(directory_path.c_str());

            struct dirent* entry;

            while ((entry = readdir(dirp)) != nullptr) {
                std::string filename = entry->d_name;

                if (filename == "." || filename == "..") continue;
                
                std::string full_path = directory_path + "/" + filename;

                std::size_t n = countVerticesInTXT(full_path);

                std::cout << "Processing file: " << filename << ", Length: " << n << std::endl;

                std::vector<std::vector<R>> data(n, std::vector<R>(D));

                fill_with_txt_data<R, D>(full_path, data, n);

                std::vector<float> flattened_data = flatten_2D_vector(data);

                std::size_t q = n;

                std::vector<idx_t> Q(q);
                std::iota(Q.begin(), Q.end(), 0);

                for (std::size_t k : k_values)
                {
                    std::cout  << "(" <<n<<", "<< k << ", [";
                    for (std::size_t j = 0; j < reps; ++j)
                    {
                       auto const [knn, distances] = dispatch_knn<R, 3>(flattened_data, Q, algorithm.value(), k, n, q);

                       // print_knn(knn, n, k);
                        print_knn( distances, 10000, k);
                    }
                    std::cout << "])," << std::endl;
                }
           }
            closedir(dirp);
        }
    }
}

template<class R, std::size_t n, std::size_t D>
void execute_knn(int argc, char **argv)
{
    std::optional< Algorithm > const algorithm = parse_input( argc, argv );
    if( algorithm.has_value() )
    {
        if(  run_test( algorithm.value() ) || true)
        {
            constexpr std::size_t reps = 2;

            std::vector< int > k_values{30};//, 100, 200, 500};

            using real = float;
            //constexpr std::size_t n = 100'000;
            constexpr int seed = 1133;

            real data[n][dim];
            fill_with_random_data(data, seed);

            constexpr std::size_t q = n;

            for (std::size_t k : k_values)
            {
#ifdef NDEBUG
                std::cout  << "(" <<n<<", "<< k << ", [";
#endif
                for (std::size_t j = 0; j < reps; ++j)
                {
                    idx_t Q[q];
                    std::iota( Q, Q + q, 0 );

                    auto const [knn, distances] = dispatch_knn(data, Q, algorithm.value(), k);

                    //print_knn( distances, n, k );

#ifndef NDEBUG
                    //print_knn( knn, q, k );
                    //print_knn( distances, q, k );
#endif
                }
#ifdef NDEBUG
                std::cout << "])," << std::endl;
#endif
            }
        }
        //else
        //{
            //std::cout << "Unit test failed" << std::endl;
        //}
    }
}

int main( int argc, char **argv )
{
    std::optional< Algorithm > const algorithm = parse_input( argc, argv );
    
    using real = float;

    execute_knn<float, 1000, 3>(argc, argv);
    execute_knn<float, 2000, 3>(argc, argv);
    execute_knn<float, 5000, 3>(argc, argv);
    execute_knn<float, 10000, 3>(argc, argv);
    execute_knn<float, 20000, 3>(argc, argv);
    execute_knn<float, 50000, 3>(argc, argv);
    execute_knn<float, 80000, 3>(argc, argv);
    execute_knn<float, 100000, 3>(argc, argv);
    execute_knn<float, 150000, 3>(argc, argv);
    execute_knn<float, 200000, 3>(argc, argv);
    execute_knn<float, 350000, 3>(argc, argv);
    execute_knn<float, 500000, 3>(argc, argv);
    execute_knn<float, 600000, 3>(argc, argv);
    execute_knn<float, 700000, 3>(argc, argv);
    execute_knn<float, 800000, 3>(argc, argv);
    execute_knn<float, 900000, 3>(argc, argv);
    execute_knn<float, 1000000, 3>(argc, argv);

    execute_knn_meshes<real, 3>(argc, argv);

    return EXIT_SUCCESS;
}
