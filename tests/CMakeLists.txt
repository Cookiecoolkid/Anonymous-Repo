# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# 定义通用库依赖
set(COMMON_LIBS
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    ${EFANNA2E_LIB}
    nsg
    pthread
    m
    ${OpenMP_CXX_FLAGS}
    aux_util
    CNNS
)

# 添加测试程序
add_executable(cluster_IVF_nndescent cluster_IVF_nndescent.cpp)
add_executable(nndescent_nsg nndescent_nsg.cpp)
add_executable(cluster_hnsw_nsg_search cluster_hnsw_nsg_search.cpp)
add_executable(cluster_hnsw_nsg_search_pipeline cluster_hnsw_nsg_search_pipeline.cpp)
add_executable(cluster_hnsw_nsg_search_add_load_time cluster_hnsw_nsg_search_add_load_time.cpp)
add_executable(cluster_hnsw_nsg_search_query cluster_hnsw_nsg_search_query.cpp)
add_executable(cluster_hnsw_nsg_search_mmap cluster_hnsw_nsg_search_mmap.cpp)
add_executable(test_index_build test_index_build.cpp)
add_executable(test_index_search test_index_search.cpp)
add_executable(test_search_mmap test_search_mmap.cpp)
add_executable(test_search_pread test_search_pread.cpp)
add_executable(test_search_serial test_search_serial.cpp)

# 链接库
foreach(TARGET cluster_IVF_nndescent nndescent_nsg cluster_hnsw_nsg_search 
        cluster_hnsw_nsg_search_pipeline cluster_hnsw_nsg_search_add_load_time 
        cluster_hnsw_nsg_search_query cluster_hnsw_nsg_search_mmap 
        test_index_build test_index_search test_search_mmap test_search_pread test_search_serial)
    target_link_libraries(${TARGET} ${COMMON_LIBS})
endforeach()

# 添加experiment_feature子目录
add_subdirectory(experiment_feature)

