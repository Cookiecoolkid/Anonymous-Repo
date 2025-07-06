set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(aux_util STATIC
    aux_util.cpp
)

# 设置库的输出目录
set_target_properties(aux_util PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 添加头文件目录
target_include_directories(aux_util
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/utils
)

# 添加编译选项
target_compile_options(aux_util PRIVATE
    -O3
    -march=native
    -Wall
    -DINFO
    -msse4.2
    -mavx2
    -mfma
)

# 添加链接库
target_link_libraries(aux_util
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

add_library(diskann_utils STATIC
    ann_exception.cpp
    linux_aligned_file_reader.cpp
    logger.cpp
    windows_aligned_file_reader.cpp
)

target_include_directories(diskann_utils PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/diskann_utils
)

target_link_libraries(diskann_utils 
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

target_compile_options(diskann_utils PRIVATE
    -O3
    -march=native
    -Wall
    -DINFO
    -msse4.2
    -mavx2
    -mfma
)
