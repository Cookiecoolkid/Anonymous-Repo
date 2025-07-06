# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加显式模板实例化
add_library(CNNS_templates STATIC
    data_load.cpp
)

# 为CNNS_templates添加PIC编译选项
set_target_properties(CNNS_templates PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(CNNS_templates PRIVATE
    CNNS_TEMPLATE_INSTANTIATION
)

target_include_directories(CNNS_templates
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/efanna_graph/include
        ${PROJECT_SOURCE_DIR}/include/diskann_utils
        ${PROJECT_SOURCE_DIR}/include/nsg
)

target_link_libraries(CNNS_templates
    PUBLIC
        ${FAISS_LIB}
        ${OpenBLAS_LIB}
        ${EFANNA2E_LIB}
        aux_util
        diskann_utils
        pthread
        m
        ${OpenMP_CXX_FLAGS}
)

# 添加源文件
set(SOURCES
    index_build.cpp
    index_search.cpp
    index_search_mmap.cpp
    index_search_pread.cpp
    index_search_serial.cpp
    data_load.cpp
    statistics.cpp
)

# 创建库目标
add_library(CNNS STATIC ${SOURCES})

# 设置库的输出目录和PIC编译选项
set_target_properties(CNNS PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)

# 添加头文件目录
target_include_directories(CNNS
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/efanna_graph/include
        ${PROJECT_SOURCE_DIR}/include/diskann_utils
        ${PROJECT_SOURCE_DIR}/include/nsg
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/faiss
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/faiss/faiss
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/OpenBLAS/include
        ${OpenBLAS_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
target_link_libraries(CNNS
    PUBLIC
        ${FAISS_LIB}
        ${OpenBLAS_LIB}
        ${EFANNA2E_LIB}
        nsg
        aux_util
        diskann_utils
        pthread
        m
        ${OpenMP_CXX_FLAGS}
)

# 添加子目录
add_subdirectory(utils)
add_subdirectory(nsg)