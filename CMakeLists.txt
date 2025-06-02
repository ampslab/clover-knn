cmake_minimum_required(VERSION 3.27)
project(gpu-knn LANGUAGES CUDA CXX)

file(GLOB CU_FILES
    src/*.cu
    include/treelogy/GPU/knearestneighbor/kdtree/gpu_non_lockstep/*.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(linear-scans ${CU_FILES})

if(LINK_FAISS)
    add_compile_definitions(USE_FAISS)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
    find_package(OpenMP REQUIRED)
    target_link_libraries(linear-scans PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(linear-scans PRIVATE -lcublas)
    target_link_libraries(linear-scans PRIVATE
    	    ${CMAKE_SOURCE_DIR}/external/faiss/build/faiss/libfaiss.a)
endif()

target_compile_options(linear-scans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -arch=sm_89;
                        -std=c++20;
                        -Xptxas="-v";
                       >)