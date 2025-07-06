#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "nsg/neighbor.h"
#include "nsg/index_nsg.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "data_load.h"
#include "statistics.h"

namespace CNNS {

template<typename T>
class IndexBuilder {
public:
    // 构造函数
    IndexBuilder(const std::string& prefix,
                int n_clusters,
                int m_centroids,
                int k_nndescent = 100,
                int l_nndescent = 100,
                int iter = 10,
                int s = 10,
                int r = 100,
                int L_nsg = 40,
                int R_nsg = 50,
                int C_nsg = 500);

    // 析构函数
    ~IndexBuilder();

    // 构建索引
    bool build(const std::string& data_file, bool use_mmap = false, Statistics* stats = nullptr);
    bool build(const std::string& data_file, DataFormat format, bool use_mmap = false, Statistics* stats = nullptr);
    bool build_auto_format(const std::string& data_file, bool use_mmap = false, Statistics* stats = nullptr);
    
    // 构建索引（内存映射版本，适用于大数据集）
    bool build_mmap(const std::string& data_file, DataFormat format, Statistics* stats = nullptr);

    // 释放资源
    void release();

private:
    // 构建IVF索引
    bool buildIVFIndex(const std::vector<float>& data, unsigned dim, unsigned points_num);

    // 构建NNDescent图（内存映射版本，不需要传入数据）
    bool buildNNDescentGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                            unsigned dim);

    // 构建NSG图
    bool buildNSGGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                      unsigned dim,
                      bool use_mmap = false);

    // 构建NSG图（带数据）
    bool buildNSGDataGraph(const std::vector<T>& data,
                          const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                          unsigned dim,
                          bool use_mmap = false);

    // 构建ClusterData
    bool buildClusterData(const std::vector<T>& data,
                         const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                         unsigned dim);

    // 构建ClusterCentroids
    bool buildCentroids(const std::vector<float>& data,
                       const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                       unsigned dim);

    // 构建NavigationHNSW
    bool buildNavigationHNSW(unsigned dim);

    // 保存NNDescent图
    bool saveNNDescentGraph(efanna2e::IndexGraph& index, 
                           faiss::idx_t cluster_id);

    // 保存NSG图
    bool saveNSG(efanna2e::IndexNSG& index,
                faiss::idx_t cluster_id,
                bool use_mmap = false);
    
    // 保存NSG图（带数据）
    bool saveNSGData(efanna2e::IndexNSG& index,
                    const std::vector<T>& data,
                    const std::vector<faiss::idx_t>& ids_in_cluster,
                    faiss::idx_t cluster_id,
                    bool use_mmap = false);

    // 保存和读取映射
    bool buildClusterMappings(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids);

    // 内存映射版本的辅助函数
    bool buildIVFIndex_mmap(const std::string& data_file, DataFormat format, unsigned dim, unsigned points_num, size_t batch_size, std::vector<faiss::idx_t>& cluster_assignments);
    bool buildClusterData_mmap(const std::string& data_file, DataFormat format, 
                              const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                              unsigned dim, unsigned points_num, size_t batch_size);

    // 共享内存映射版本的辅助函数（避免重复映射）
    bool buildIVFIndex_mmap_shared(const std::string& data_file, DataFormat format, unsigned dim, unsigned points_num, size_t batch_size, std::vector<faiss::idx_t>& cluster_assignments, const void* shared_data_ptr);
    bool buildClusterData_mmap_shared(const std::string& data_file, DataFormat format, 
                                     const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                     unsigned dim, unsigned points_num, size_t batch_size, const void* shared_data_ptr);

    bool buildCentroidsFromIVF(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                              unsigned dim);

    // 成员变量
    std::string prefix_;
    int n_clusters_;
    int m_centroids_;
    int k_nndescent_;
    int l_nndescent_;
    int iter_;
    int s_;
    int r_;
    int L_nsg_;
    int R_nsg_;
    int C_nsg_;

    // 索引相关指针
    std::unique_ptr<faiss::IndexIVFFlat> index_ivf_;
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::unique_ptr<faiss::IndexHNSWFlat> index_hnsw_;
};
} // namespace CNNS
