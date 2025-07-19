# GANNS 🔥
GANNS (Graph Abstraction Nearest Neighbor Search) is a high-performance approximate nearest neighbor search framework built upon graph-based indexing techniques. It supports billion-scale vector data and is optimized for memory and performance on large-scale search problems.

## Features 🌟
- Hybrid graph indexing 
- Out-of-core support for large datasets
- Compatible with multiple vector format (.fvecs, .bvecs)
- Designed for high recall and fast query latency at scale

## How To Compile ⌛️
### 1. Requirements
Before compiling GANNS, make sure the following software and libraries are installed:

- C++17 compatible compiler (e.g., g++ $\geq$ 7)
- CMake ($\geq$ 3.10)
- OpenMP (version 2.0+)
- BLAS library (Intel MKL recommended)
- Boost library ($\geq$ 1.65)
- libaio (for async I/O support on Linux)

To install most dependencies on Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  libboost-all-dev \
  libopenblas-dev \
  libaio-dev \
  libomp-dev
```

### 2. Compile Dependencies
Efanna Graph
GANNS relies on Efanna Graph, which includes a customized version of Faiss. First compile it:

The detailed guide of [Efanna Graph GitHub](https://github.com/ZJULearning/efanna_graph).

Compile Faiss inside Efanna first:
```bash
cd extern_libraries/faiss
```
then follow the guide of faiss to compile it: [FAISS GitHub](https://github.com/facebookresearch/faiss).

then back to efanna_graph root
```bash
cmake .
make -j4
```
### 3. Compile GANNS


```bash
cd GANNS
mkdir build && cd build
cmake ..
make -j4
```



## Usage 🔨 

### Build Index
```bash
cd build
./tests/test_index_build <data_file> <data_format> <n_clusters> <m_centroids> <K> <L> <iter> <S> <R> <L_nsg> <R_nsg> <C_nsg> <prefix>
Usage: ./test_index_build <data_file> <data_format> <n_clusters> <m_centroids> <K> <L> <iter> <S> <R> <L_nsg> <R_nsg> <C_nsg> <prefix>
  data_format: data file format (fvecs or bvecs)
  K: number of neighbors in NNDescent graph (default: 100)
  L: number of candidates in NNDescent graph (default: 100)
  iter: number of iterations (default: 10)
  S: size of candidate set (default: 10)
  R: maximum degree of each node (default: 100)
  L_nsg: number of candidates in NSG (default: 20)
  R_nsg: maximum degree of each node in NSG (default: 20)
  C_nsg: number of candidates in NSG (default: 500)
  prefix: prefix directory for output files
```

### Search on Index
```bash
cd build
./tests/test_search_mmap 
Usage: ./test_search_mmap <path_to_query_data> <query_format> <path_to_ground_truth> <nprobe> <search_K> <search_L> <prefix> <num_threads>
  query_format: query data file format (fvecs or bvecs)
  nprobe: number of clusters to search (default: 50)
  search_K: number of neighbors to search in NSG (default: 100)
  search_L: number of candidates in NSG search (default: 100 / should >= search_K)
  prefix: directory prefix for all data files
  num_threads: number of threads to use (-1 for auto-detect)
```

### Data Format Supported 🔢
GANNS supports the following vector data formats:
- **fvecs**: Float vectors
- **bvecs**: Byte vectors
- **ivecs**: Integer vectors

Detailed format see [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)


# License 📒
MIT Licensed
This project inherits licenses from FAISS and EfannaGraph. Please check each subproject’s license for details.