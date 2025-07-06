#include "index_build.h"
#include "aux_util.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <chrono>
#include <iomanip>
#include <omp.h>

namespace CNNS {

template<typename T>
IndexBuilder<T>::IndexBuilder(const std::string& prefix,
                            int n_clusters,
                            int m_centroids,
                            int k_nndescent,
                            int l_nndescent,
                            int iter,
                            int s,
                            int r,
                            int L_nsg,
                            int R_nsg,
                            int C_nsg)
    : prefix_(prefix),
      n_clusters_(n_clusters),
      m_centroids_(m_centroids),
      k_nndescent_(k_nndescent),
      l_nndescent_(l_nndescent),
      iter_(iter),
      s_(s),
      r_(r),
      L_nsg_(L_nsg),
      R_nsg_(R_nsg),
      C_nsg_(C_nsg) {
}

template<typename T>
IndexBuilder<T>::~IndexBuilder() {
    release();
}

template<typename T>
void IndexBuilder<T>::release() {
    // 释放所有智能指针管理的资源
    quantizer_.reset();
    index_ivf_.reset();
    index_hnsw_.reset();
}

template<typename T>
bool IndexBuilder<T>::build(const std::string& data_file, bool use_mmap, Statistics* stats) {
    // 默认使用FVECS格式
    return build(data_file, DataFormat::FVECS, use_mmap, stats);
}

template<typename T>
bool IndexBuilder<T>::build_auto_format(const std::string& data_file, bool use_mmap, Statistics* stats) {
    // 自动检测文件格式
    DataFormat format = detect_file_format(data_file);
    std::cout << "Detected format: " << (format == DataFormat::FVECS ? "FVECS" : 
                                        format == DataFormat::BVECS ? "BVECS" : "IVECS") << std::endl;
    return build(data_file, format, use_mmap, stats);
}

template<typename T>
bool IndexBuilder<T>::build(const std::string& data_file, DataFormat format, bool use_mmap, Statistics* stats) {
    // 创建必要的目录
    std::filesystem::create_directories(prefix_);
    std::filesystem::create_directories(prefix_ + "/cluster_data");
    std::filesystem::create_directories(prefix_ + "/nndescent");
    std::filesystem::create_directories(prefix_ + "/nsg_graph");
    std::filesystem::create_directories(prefix_ + "/mapping");

    std::vector<T> data;
    std::vector<float> float_data; // 用于FAISS的float数据
    unsigned dim = 0;
    unsigned points_num = 0;

    // 根据格式加载数据，使用移动语义避免双重内存占用
    switch (format) {
        case DataFormat::FVECS: {
            auto loaded_float_data = CNNS::load_fvecs(data_file, points_num, dim);
            data.assign(loaded_float_data.begin(), loaded_float_data.end());
            // 对于FVECS，直接使用float数据
            float_data = std::move(loaded_float_data);
            break;
        }
        case DataFormat::BVECS: {
            auto bvec_data = CNNS::load_bvecs(data_file, points_num, dim);
            data.assign(bvec_data.begin(), bvec_data.end());
            // 将uint8转换为float
            float_data.resize(static_cast<size_t>(points_num) * static_cast<size_t>(dim));
            for (size_t i = 0; i < static_cast<size_t>(points_num) * static_cast<size_t>(dim); ++i) {
                float_data[i] = static_cast<float>(bvec_data[i]);
            }
            break;
        }
        case DataFormat::IVECS: {
            auto int_data = CNNS::load_ivecs(data_file, points_num, dim);
            data.assign(int_data.begin(), int_data.end());
            // 将int转换为float
            float_data.resize(static_cast<size_t>(points_num) * static_cast<size_t>(dim));
            for (size_t i = 0; i < static_cast<size_t>(points_num) * static_cast<size_t>(dim); ++i) {
                float_data[i] = static_cast<float>(int_data[i]);
            }
            break;
        }
        default:
            std::cerr << "Unsupported data format" << std::endl;
            return false;
    }
    

    std::cout << "Loaded " << points_num << " points with dimension " << dim 
              << " (memory usage: " << (data.size() * sizeof(T) / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;

    // 统计总时间
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 统计IVF构建时间
    auto ivf_start = std::chrono::high_resolution_clock::now();
    
    // 构建IVF索引（使用float数据）
    if (!buildIVFIndex(float_data, dim, points_num)) {
        std::cerr << "Failed to build IVF index" << std::endl;
        return false;
    }

    // 获取聚类分配（使用float数据）
    std::vector<faiss::idx_t> cluster_assignments(points_num);
    index_ivf_->quantizer->assign(points_num, float_data.data(), cluster_assignments.data());
    // 构建cluster到点id的映射（用于后续处理）
    std::map<faiss::idx_t, std::vector<faiss::idx_t>> cluster_to_ids;
    for (size_t i = 0; i < points_num; ++i) {
        cluster_to_ids[cluster_assignments[i]].push_back(i);
    }

    std::cout << "Build cluster to ids mapping successfully" << std::endl;
    
    index_ivf_->make_direct_map();

    // 构建质心（使用原始数据类型）
    if (!buildCentroids(float_data, cluster_to_ids, dim)) {
        std::cerr << "Failed to build centroids" << std::endl;
        return false;
    }
    
    auto ivf_end = std::chrono::high_resolution_clock::now();
    auto ivf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ivf_end - ivf_start);
    double ivf_time_ms = ivf_duration.count();
    std::cout << "IVF build time: " << ivf_time_ms / 1000.0 << " s" << std::endl;
    
    // 记录IVF构建时间到统计对象
    if (stats) {
        stats->record_ivf_build_time(ivf_time_ms);
    }

    // 统计HNSW构建时间
    auto hnsw_start = std::chrono::high_resolution_clock::now();
    
    // 构建导航HNSW索引
    if (!buildNavigationHNSW(dim)) {
        std::cerr << "Failed to build navigation HNSW" << std::endl;
        return false;
    }
    
    auto hnsw_end = std::chrono::high_resolution_clock::now();
    auto hnsw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(hnsw_end - hnsw_start);
    double hnsw_time_ms = hnsw_duration.count();
    std::cout << "HNSW build time: " << hnsw_time_ms / 1000.0 << " s" << std::endl;
    
    // 记录HNSW构建时间到统计对象
    if (stats) {
        stats->record_hnsw_build_time(hnsw_time_ms);
    }

    
    // 构建并保存聚类映射
    if (!buildClusterMappings(cluster_to_ids)) {
        std::cerr << "Failed to build cluster mappings" << std::endl;
        return false;
    }

    // 构建聚类数据（使用原始数据类型）
    if (!buildClusterData(data, cluster_to_ids, dim)) {
        std::cerr << "Failed to build cluster data" << std::endl;
        return false;
    }

    auto nndescent_start = std::chrono::high_resolution_clock::now();

    // 构建NNDescent图（使用原始数据类型）
    if (!buildNNDescentGraph(cluster_to_ids, dim)) {
        std::cerr << "Failed to build NNDescent graph" << std::endl;
        return false;
    }

    auto nndescent_end = std::chrono::high_resolution_clock::now();
    auto nndescent_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nndescent_end - nndescent_start);
    double nndescent_time_ms = nndescent_duration.count();
    std::cout << "NNDescent build time: " << nndescent_time_ms / 1000.0 << " s" << std::endl;
    
    // 记录NNDescent构建时间到统计对象
    if (stats) {
        stats->record_nndescent_build_time(nndescent_time_ms);
    }

    // 统计NSG构建时间
    auto nsg_start = std::chrono::high_resolution_clock::now();

    // 构建NSG图
    if (!buildNSGGraph(cluster_to_ids, dim, use_mmap)) {
        std::cerr << "Failed to build NSG graph" << std::endl;
        return false;
    }
    
    auto nsg_end = std::chrono::high_resolution_clock::now();
    auto nsg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nsg_end - nsg_start);
    double nsg_time_ms = nsg_duration.count();
    std::cout << "NSG build time: " << nsg_time_ms / 1000.0 << " s" << std::endl;
    
    // 记录NSG构建时间到统计对象
    if (stats) {
        stats->record_nsg_build_time(nsg_time_ms);
    }

    // 统计总时间
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    double total_time_ms = total_duration.count();
    std::cout << "Total build time: " << total_time_ms / 1000.0 << " s" << std::endl;
    
    // 记录总构建时间到统计对象
    if (stats) {
        stats->record_total_build_time(total_time_ms);
        stats->calculate_build_percentages();
    }
    
    // 输出时间占比
    double ivf_percent = (double)ivf_duration.count() / total_duration.count() * 100;
    double hnsw_percent = (double)hnsw_duration.count() / total_duration.count() * 100;
    double nndescent_percent = (double)nndescent_duration.count() / total_duration.count() * 100;
    double nsg_percent = (double)nsg_duration.count() / total_duration.count() * 100;
    
    std::cout << "Time ratio - IVF: " << std::fixed << std::setprecision(1) << ivf_percent 
              << "%, HNSW: " << hnsw_percent << "%, NNDescent: " << nndescent_percent 
              << "%, NSG: " << nsg_percent << "%" << std::endl;

    return true;
}

template<typename T>
bool IndexBuilder<T>::buildIVFIndex(const std::vector<float>& data, unsigned dim, unsigned points_num) {
    try {
        // 创建量化器
        quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim);

        // 创建IVF索引
        index_ivf_ = std::make_unique<faiss::IndexIVFFlat>(quantizer_.get(), dim, n_clusters_, faiss::METRIC_L2);

        // 训练IVF索引
        std::vector<faiss::idx_t> ids(points_num);
        std::iota(ids.begin(), ids.end(), 0);
        index_ivf_->train(points_num, data.data());
        index_ivf_->add_with_ids(points_num, data.data(), ids.data());

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building IVF index: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildClusterData(const std::vector<T>& data,
                                     const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                     unsigned dim) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            std::string cluster_file = prefix_ + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".data";
            std::ofstream out(cluster_file, std::ios::binary);
            if (!out.is_open()) {
                std::cerr << "Cannot open cluster file " << cluster_file << std::endl;
                return false;
            }

            // 写入每个向量的维度
            // out.write((char*)&dim, sizeof(dim));
            for (faiss::idx_t id : ids_in_cluster) {
                // out.write((char*)&dim, sizeof(dim));
                out.write((char*)(data.data() + id * dim), dim * sizeof(T));
            }
            out.close();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building cluster data: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildCentroids(const std::vector<float>& data,
                                   const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                   unsigned dim) {
    try {
        // 提取质心并保存
        std::vector<float> centroids((n_clusters_ * (m_centroids_ + 1)) * dim);
        std::random_device rd;
        std::mt19937 gen(rd());

        // 保存质心文件头信息
        std::string centroids_path = prefix_ + "/centroids.data";
        std::ofstream centroids_file(centroids_path, std::ios::binary);
        centroids_file.write((char*)&n_clusters_, sizeof(n_clusters_));
        centroids_file.write((char*)&m_centroids_, sizeof(m_centroids_));
        centroids_file.write((char*)&dim, sizeof(dim));

        for (int i = 0; i < n_clusters_; i++) {
            // 保存质心
            index_ivf_->reconstruct(i, centroids.data() + i * (m_centroids_ + 1) * dim);
            centroids_file.write((char*)&dim, sizeof(dim));
            centroids_file.write((char*)(centroids.data() + i * (m_centroids_ + 1) * dim), dim * sizeof(float));

            // 随机选择m个点
            const auto& ids_in_cluster = cluster_to_ids.at(i);
            if ((int)ids_in_cluster.size() > m_centroids_) {
                std::vector<size_t> indices(ids_in_cluster.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), gen);
                
                for (int j = 0; j < m_centroids_; j++) {
                    size_t idx = indices[j];
                    memcpy(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim,
                           data.data() + ids_in_cluster[idx] * dim,
                           dim * sizeof(float));
                    centroids_file.write((char*)&dim, sizeof(dim));
                    centroids_file.write((char*)(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim), dim * sizeof(float));
                }
            } else {
                // 如果cluster中的点数不足m，则重复使用已有点
                for (int j = 0; j < m_centroids_; j++) {
                    size_t idx = j % ids_in_cluster.size();
                    memcpy(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim,
                           data.data() + ids_in_cluster[idx] * dim,
                           dim * sizeof(float));
                    centroids_file.write((char*)&dim, sizeof(dim));
                    centroids_file.write((char*)(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim), dim * sizeof(float));
                }
            }
        }
        centroids_file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building centroids: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNavigationHNSW(unsigned dim) {
    try {
        // 读取质心文件
        std::string centroids_path = prefix_ + "/centroids.data";
        std::ifstream centroids_file(centroids_path, std::ios::binary);
        if (!centroids_file.is_open()) {
            std::cerr << "Cannot open centroids file for reading" << std::endl;
            return false;
        }

        // 读取文件头信息
        int n_clusters, m_centroids;
        centroids_file.read((char*)&n_clusters, sizeof(n_clusters));
        centroids_file.read((char*)&m_centroids, sizeof(m_centroids));
        unsigned file_dim;
        centroids_file.read((char*)&file_dim, sizeof(file_dim));

        if (n_clusters != n_clusters_ || m_centroids != m_centroids_ || file_dim != dim) {
            std::cerr << "Centroids file format mismatch" << std::endl;
            return false;
        }

        // 读取所有质心数据（使用float类型）
        std::vector<float> centroids(n_clusters_ * (m_centroids_ + 1) * dim);
        for (int i = 0; i < n_clusters_ * (m_centroids_ + 1); i++) {
            unsigned vec_dim;
            centroids_file.read((char*)&vec_dim, sizeof(vec_dim));
            if (vec_dim != dim) {
                std::cerr << "Vector dimension mismatch" << std::endl;
                return false;
            }
            centroids_file.read((char*)(centroids.data() + i * dim), dim * sizeof(float));
        }
        centroids_file.close();

        // 创建HNSW索引并保存
        index_hnsw_ = std::make_unique<faiss::IndexHNSWFlat>(dim, 32, faiss::METRIC_L2);
        index_hnsw_->add(n_clusters_ * (m_centroids_ + 1), centroids.data());
        faiss::write_index(index_hnsw_.get(), (prefix_ + "/hnsw_memory.index").c_str());

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building navigation HNSW: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildClusterMappings(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids) {
    try {
        // 保存每个聚类的映射
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            std::string mapping_filename = prefix_ + "/mapping/mapping_" + std::to_string(cluster_id);
            std::ofstream mapping_file(mapping_filename, std::ios::binary);
            if (!mapping_file.is_open()) {
                std::cerr << "Cannot open mapping file " << mapping_filename << std::endl;
                return false;
            }
            // 首先写入points_num
            unsigned points_num = ids_in_cluster.size();
            mapping_file.write((char*)&points_num, sizeof(unsigned));
            // 然后写入mapping数据
            mapping_file.write((char*)ids_in_cluster.data(), ids_in_cluster.size() * sizeof(faiss::idx_t));
            mapping_file.close();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building cluster mappings: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNNDescentGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                            unsigned dim) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            // 创建并加载聚类数据
            CNNS::ClusterMMap cluster_data;
            if (!load_cluster_data_mmap(cluster_id, dim, cluster_data, prefix_)) {
                std::cerr << "Failed to load cluster data for cluster " << cluster_id << std::endl;
                return false;
            }

            // 构建NNDescent图
            efanna2e::IndexRandom init_index(dim, ids_in_cluster.size());
            efanna2e::IndexGraph index(dim, ids_in_cluster.size(), efanna2e::L2, &init_index);
            efanna2e::Parameters paras;
            paras.Set<unsigned>("K", k_nndescent_);
            paras.Set<unsigned>("L", l_nndescent_);
            paras.Set<unsigned>("iter", iter_);
            paras.Set<unsigned>("S", s_);
            paras.Set<unsigned>("R", r_);

            index.Build(ids_in_cluster.size(), cluster_data.data_ptr, paras);

            // 保存NNDescent图
            if (!saveNNDescentGraph(index, cluster_id)) {
                std::cerr << "Failed to save NNDescent graph for cluster " << cluster_id << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building NNDescent graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveNNDescentGraph(efanna2e::IndexGraph& index, 
                                       faiss::idx_t cluster_id) {
    try {
        std::string graph_filename = prefix_ + "/nndescent/nndescent_" + 
                                   std::to_string(cluster_id) + ".graph";
        index.Save(graph_filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NNDescent graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNSGGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                  unsigned dim,
                                  bool use_mmap) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            // 创建并加载聚类数据
            CNNS::ClusterMMap cluster_data;
            if (!load_cluster_data_mmap(cluster_id, dim, cluster_data, prefix_)) {
                std::cerr << "Failed to load cluster data for cluster " << cluster_id << std::endl;
                return false;
            }
            
            // 构建NSG
            efanna2e::IndexNSG index(dim, ids_in_cluster.size(), efanna2e::L2, nullptr);
            efanna2e::Parameters paras;
            paras.Set<unsigned>("L", L_nsg_);
            paras.Set<unsigned>("R", R_nsg_);
            paras.Set<unsigned>("C", C_nsg_);
            paras.Set<std::string>("nn_graph_path", 
                prefix_ + "/nndescent/nndescent_" + std::to_string(cluster_id) + ".graph");

            index.Build(ids_in_cluster.size(), cluster_data.data_ptr, paras);

            // 保存NSG
            if (!saveNSG(index, cluster_id, use_mmap)) {
                std::cerr << "Failed to save NSG for cluster " << cluster_id << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building NSG graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveNSG(efanna2e::IndexNSG& index,
                            faiss::idx_t cluster_id,
                            bool use_mmap) {
    try {
        std::string nsg_filename = prefix_ + "/nsg_graph/nsg_" + 
                                 std::to_string(cluster_id) + ".nsg";
        if (use_mmap) {
            index.Save_mmap_with_dist(nsg_filename.c_str());
        } else {
            index.Save(nsg_filename.c_str());
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NSG: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNSGDataGraph(const std::vector<T>& data,
                                      const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                      unsigned dim,
                                      bool use_mmap) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            // 构建NSG
            efanna2e::IndexNSG index(dim, ids_in_cluster.size(), efanna2e::L2, nullptr);
            efanna2e::Parameters paras;
            paras.Set<unsigned>("L", L_nsg_);
            paras.Set<unsigned>("R", R_nsg_);
            paras.Set<unsigned>("C", C_nsg_);
            paras.Set<std::string>("nn_graph_path", 
                prefix_ + "/nndescent/nndescent_" + std::to_string(cluster_id) + ".graph");

            // 为每个点分配数据
            std::vector<float> cluster_data(ids_in_cluster.size() * dim);
            for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
                memcpy(cluster_data.data() + i * dim,
                       data.data() + ids_in_cluster[i] * dim,
                       dim * sizeof(T));
            }

            index.Build(ids_in_cluster.size(), cluster_data.data(), paras);

            // 保存NSG（带数据）
            if (!saveNSGData(index, data, ids_in_cluster, cluster_id, use_mmap)) {
                std::cerr << "Failed to save NSG with data for cluster " << cluster_id << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building NSG graph with data: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveNSGData(efanna2e::IndexNSG& index,
                                const std::vector<T>& data,
                                const std::vector<faiss::idx_t>& ids_in_cluster,
                                faiss::idx_t cluster_id,
                                bool use_mmap) {
    try {
        // 1. 保存图结构到 nsg_graph 目录
        std::string nsg_filename = prefix_ + "/nsg_graph/nsg_" + 
                                 std::to_string(cluster_id) + ".nsg";
        index.Save_mmap_with_dist(nsg_filename.c_str());

        // 2. 保存数据到 nsg_data 目录
        std::string nsg_data_filename = prefix_ + "/nsg_data/nsg_" + 
                                      std::to_string(cluster_id) + ".data";
        // 确保目录存在
        std::filesystem::create_directories(prefix_ + "/nsg_data");
        index.Save_with_data(nsg_data_filename.c_str());

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NSG with data: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::build_mmap(const std::string& data_file, DataFormat format, Statistics* stats) {
    // 创建必要的目录
    std::filesystem::create_directories(prefix_);
    std::filesystem::create_directories(prefix_ + "/cluster_data");
    std::filesystem::create_directories(prefix_ + "/nndescent");
    std::filesystem::create_directories(prefix_ + "/nsg_graph");
    std::filesystem::create_directories(prefix_ + "/mapping");

    unsigned dim = 0;
    unsigned points_num = 0;
    
    // 获取文件信息
    std::ifstream file(data_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << data_file << std::endl;
        return false;
    }
    
    // 读取维度信息
    file.read((char*)&dim, 4);
    file.close();
    
    // 计算点数
    std::filesystem::path file_path(data_file);
    size_t file_size = std::filesystem::file_size(file_path);
    
    switch (format) {
        case DataFormat::FVECS:
            points_num = file_size / ((dim + 1) * 4);
            break;
        case DataFormat::BVECS:
            points_num = file_size / (4 + dim);
            break;
        case DataFormat::IVECS:
            points_num = file_size / ((dim + 1) * 4);
            break;
        default:
            std::cerr << "Unsupported data format" << std::endl;
            return false;
    }
    
    // 统一设置 batch_size
    const size_t batch_size = points_num >= 1e8 ? static_cast<size_t>(points_num * 0.01) : 
                              points_num >= 1e7 ? static_cast<size_t>(points_num * 0.05) : 
                              points_num * 0.5;
    
    std::cout << "Processing " << points_num << " points with dimension " << dim 
              << " using memory mapping (file size: " << (file_size / (1024.0 * 1024.0 * 1024.0)) << " GB)"
              << " (batch_size: " << batch_size << ")" << std::endl;
    
    // 创建一次内存映射，供整个构建过程使用
    MMapResource shared_mmap_res;
    const void* shared_data_ptr = nullptr;
    size_t shared_mapped_points = 0;
    
    switch (format) {
        case DataFormat::BVECS: {
            auto [temp_mmap_res, temp_data_ptr, temp_mapped_points] = mmap_bvecs_managed(data_file);
            shared_mmap_res = std::move(temp_mmap_res);
            shared_data_ptr = temp_data_ptr;
            shared_mapped_points = temp_mapped_points;
            break;
        }
        case DataFormat::FVECS: {
            auto [temp_mmap_res, temp_data_ptr, temp_mapped_points] = mmap_fvecs_managed(data_file);
            shared_mmap_res = std::move(temp_mmap_res);
            shared_data_ptr = temp_data_ptr;
            shared_mapped_points = temp_mapped_points;
            break;
        }
        case DataFormat::IVECS: {
            auto [temp_mmap_res, temp_data_ptr, temp_mapped_points] = mmap_ivecs_managed(data_file);
            shared_mmap_res = std::move(temp_mmap_res);
            shared_data_ptr = temp_data_ptr;
            shared_mapped_points = temp_mapped_points;
            break;
        }
        default:
            std::cerr << "Unsupported data format" << std::endl;
            return false;
    }
    
    if (shared_mapped_points != points_num) {
        std::cerr << "Points number mismatch in mmap" << std::endl;
        return false;
    }
    
    // 统计总时间
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 构建IVF索引（使用共享内存映射）
    auto ivf_start = std::chrono::high_resolution_clock::now();
    
    // 获取聚类分配（使用共享内存映射）
    std::vector<faiss::idx_t> cluster_assignments(points_num);

    std::cout << "Allocate memory for cluster_assignments successfully" << std::endl;
    
    if (!buildIVFIndex_mmap_shared(data_file, format, dim, points_num, batch_size, cluster_assignments, shared_data_ptr)) {
        std::cerr << "Failed to build IVF index" << std::endl;
        return false;
    }
    
    auto ivf_end = std::chrono::high_resolution_clock::now();
    auto ivf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ivf_end - ivf_start);
    double ivf_time_ms = ivf_duration.count();
    std::cout << "IVF build time: " << ivf_time_ms / 1000.0 << " s" << std::endl;
    
    if (stats) {
        stats->record_ivf_build_time(ivf_time_ms);
    }
    
    std::cout << "Assign cluster assignments successfully" << std::endl;
    
    // 构建cluster到点id的映射
    std::map<faiss::idx_t, std::vector<faiss::idx_t>> cluster_to_ids;
    for (size_t i = 0; i < points_num; ++i) {
        cluster_to_ids[cluster_assignments[i]].push_back(i);
    }

    std::cout << "Build cluster to ids mapping successfully" << std::endl;
    
    index_ivf_->make_direct_map();

    std::cout << "Make direct map successfully" << std::endl;
    
    // 构建质心（使用已有的IVF索引）
    auto centroids_start = std::chrono::high_resolution_clock::now();
    
    if (!buildCentroidsFromIVF(cluster_to_ids, dim)) {
        std::cerr << "Failed to build centroids" << std::endl;
        return false;
    }
    
    auto centroids_end = std::chrono::high_resolution_clock::now();
    auto centroids_duration = std::chrono::duration_cast<std::chrono::milliseconds>(centroids_end - centroids_start);
    std::cout << "Centroids build time: " << centroids_duration.count() / 1000.0 << " s" << std::endl;
    
    // 构建导航HNSW索引
    auto hnsw_start = std::chrono::high_resolution_clock::now();
    
    if (!buildNavigationHNSW(dim)) {
        std::cerr << "Failed to build navigation HNSW" << std::endl;
        return false;
    }
    
    auto hnsw_end = std::chrono::high_resolution_clock::now();
    auto hnsw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(hnsw_end - hnsw_start);
    double hnsw_time_ms = hnsw_duration.count();
    std::cout << "HNSW build time: " << hnsw_time_ms / 1000.0 << " s" << std::endl;
    
    if (stats) {
        stats->record_hnsw_build_time(hnsw_time_ms);
    }
    
    // 构建并保存聚类映射
    if (!buildClusterMappings(cluster_to_ids)) {
        std::cerr << "Failed to build cluster mappings" << std::endl;
        return false;
    }
    
    // 构建聚类数据（使用共享内存映射）
    auto cluster_data_start = std::chrono::high_resolution_clock::now();
    
    if (!buildClusterData_mmap_shared(data_file, format, cluster_to_ids, dim, points_num, batch_size, shared_data_ptr)) {
        std::cerr << "Failed to build cluster data" << std::endl;
        return false;
    }
    
    auto cluster_data_end = std::chrono::high_resolution_clock::now();
    auto cluster_data_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_data_end - cluster_data_start);
    std::cout << "Cluster data build time: " << cluster_data_duration.count() / 1000.0 << " s" << std::endl;
    
    // 构建NNDescent图
    auto nndescent_start = std::chrono::high_resolution_clock::now();
    
    if (!buildNNDescentGraph(cluster_to_ids, dim)) {
        std::cerr << "Failed to build NNDescent graph" << std::endl;
        return false;
    }
    
    auto nndescent_end = std::chrono::high_resolution_clock::now();
    auto nndescent_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nndescent_end - nndescent_start);
    double nndescent_time_ms = nndescent_duration.count();
    std::cout << "NNDescent build time: " << nndescent_time_ms / 1000.0 << " s" << std::endl;
    
    if (stats) {
        stats->record_nndescent_build_time(nndescent_time_ms);
    }
    
    // 构建NSG图
    auto nsg_start = std::chrono::high_resolution_clock::now();
    
    if (!buildNSGGraph(cluster_to_ids, dim, true)) {
        std::cerr << "Failed to build NSG graph" << std::endl;
        return false;
    }
    
    auto nsg_end = std::chrono::high_resolution_clock::now();
    auto nsg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nsg_end - nsg_start);
    double nsg_time_ms = nsg_duration.count();
    std::cout << "NSG build time: " << nsg_time_ms / 1000.0 << " s" << std::endl;
    
    if (stats) {
        stats->record_nsg_build_time(nsg_time_ms);
    }
    
    // 统计总时间
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    double total_time_ms = total_duration.count();
    std::cout << "Total build time: " << total_time_ms / 1000.0 << " s" << std::endl;
    
    if (stats) {
        stats->record_total_build_time(total_time_ms);
        stats->calculate_build_percentages();
    }

    // 删除 NNDescent 图
    std::filesystem::remove_all(prefix_ + "/nndescent");
    
    // 输出时间占比
    double ivf_percent = (double)ivf_duration.count() / total_duration.count() * 100;
    double hnsw_percent = (double)hnsw_duration.count() / total_duration.count() * 100;
    double nndescent_percent = (double)nndescent_duration.count() / total_duration.count() * 100;
    double nsg_percent = (double)nsg_duration.count() / total_duration.count() * 100;
    
    std::cout << "Time ratio - IVF: " << std::fixed << std::setprecision(1) << ivf_percent 
              << "%, HNSW: " << hnsw_percent << "%, NNDescent: " << nndescent_percent 
              << "%, NSG: " << nsg_percent << "%" << std::endl;
    
    // shared_mmap_res 在函数结束时自动析构，只释放一次内存映射
    return true;
}

template<typename T>
bool IndexBuilder<T>::buildIVFIndex_mmap_shared(const std::string& data_file, DataFormat format, unsigned dim, unsigned points_num, size_t batch_size, std::vector<faiss::idx_t>& cluster_assignments, const void* shared_data_ptr) {
    try {
        // 创建量化器
        quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim);

        // 创建IVF索引
        index_ivf_ = std::make_unique<faiss::IndexIVFFlat>(quantizer_.get(), dim, n_clusters_, faiss::METRIC_L2);

        // 根据格式选择训练数据大小
        const size_t train_size = std::min(size_t(5000000), size_t(points_num)); // 500万训练点
        std::vector<float> train_data(train_size * dim);
        
        // 使用传入的共享内存映射，避免重复映射
        const void* data_ptr = shared_data_ptr;
        
        // 转换训练数据
        convert_format_to_float_training<T>(data_ptr, format, train_size, dim, train_data);

        // 训练IVF索引
        index_ivf_->train(train_size, train_data.data());
        
        // 清理训练数据，释放内存
        train_data.clear();
        train_data.shrink_to_fit();
        
        // 分批添加数据到IVF索引
        std::vector<float> batch_data(batch_size * dim);
        std::vector<faiss::idx_t> batch_ids(batch_size);
        
        for (size_t start = 0; start < points_num; start += batch_size) {
            size_t end = std::min(start + batch_size, static_cast<size_t>(points_num));
            size_t num = end - start;
            
            // 准备batch IDs（确保ID连续性）
            for (size_t i = 0; i < num; ++i) {
                batch_ids[i] = start + i;
            }

            // 根据格式转换batch数据，使用OpenMP并行化
            convert_format_to_float_batch<T>(data_ptr, format, start, num, dim, batch_data);

            // 添加batch到IVF索引
            index_ivf_->add_with_ids(num, batch_data.data(), batch_ids.data());
            
            // BUG：Can't clear here, vector will be re-used in next batch
            // batch_data.clear();
            // batch_data.shrink_to_fit();
            
            // 改进进度显示逻辑
            std::cout << "Added " << (start + num) << "/" << points_num << " points to IVF index" << std::endl;
        }
        
        // 现在使用同一个内存映射获取聚类分配
        std::cout << "Getting cluster assignments using existing memory mapping..." << std::endl;
        
        // 获取聚类分配（使用内存映射的数据）
        std::vector<float> float_data(points_num * dim);
        convert_format_to_float_batch<T>(data_ptr, format, 0, points_num, dim, float_data);
        
        // 获取聚类分配
        cluster_assignments.resize(points_num);
        index_ivf_->quantizer->assign(points_num, float_data.data(), cluster_assignments.data());
        
        std::cout << "Successfully built IVF index and got cluster assignments with " << points_num << " points" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error building IVF index with mmap: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildClusterData_mmap_shared(const std::string& data_file, DataFormat format, 
                                                  const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                                  unsigned dim, unsigned points_num, size_t batch_size, const void* shared_data_ptr) {
    try {
        // 使用传入的共享内存映射，避免重复映射
        const void* data_ptr = shared_data_ptr;
        
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            std::string cluster_file = prefix_ + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".data";
            std::ofstream out(cluster_file, std::ios::binary);
            if (!out.is_open()) {
                std::cerr << "Cannot open cluster file " << cluster_file << std::endl;
                return false;
            }

            // 根据格式转换数据到临时缓冲区
            std::vector<T> temp_data(ids_in_cluster.size() * dim);
            convert_format_to_original_batch<T>(data_ptr, format, ids_in_cluster, dim, temp_data);
            
            // 写入转换后的数据
            out.write((char*)temp_data.data(), temp_data.size() * sizeof(T));
            
            // 清理临时数据，释放内存
            temp_data.clear();
            temp_data.shrink_to_fit();
            
            out.close();
        }
        
        // 不再需要手动释放内存映射，由调用者管理
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error building cluster data with mmap: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildCentroidsFromIVF(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                          unsigned dim) {
    try {
        // 提取质心并保存
        std::vector<float> centroids((n_clusters_ * (m_centroids_ + 1)) * dim);
        std::random_device rd;
        std::mt19937 gen(rd());

        // 保存质心文件头信息
        std::string centroids_path = prefix_ + "/centroids.data";
        std::ofstream centroids_file(centroids_path, std::ios::binary);
        centroids_file.write((char*)&n_clusters_, sizeof(n_clusters_));
        centroids_file.write((char*)&m_centroids_, sizeof(m_centroids_));
        centroids_file.write((char*)&dim, sizeof(dim));

        for (int i = 0; i < n_clusters_; i++) {
            // 保存质心（从IVF索引中获取）
            index_ivf_->reconstruct(i, centroids.data() + i * (m_centroids_ + 1) * dim);
            centroids_file.write((char*)&dim, sizeof(dim));
            centroids_file.write((char*)(centroids.data() + i * (m_centroids_ + 1) * dim), dim * sizeof(float));

            // 随机选择m个点（从聚类中）
            const auto& ids_in_cluster = cluster_to_ids.at(i);
            if ((int)ids_in_cluster.size() > m_centroids_) {
                std::vector<size_t> indices(ids_in_cluster.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), gen);
                
                for (int j = 0; j < m_centroids_; j++) {
                    size_t idx = indices[j];
                    // 从IVF索引中重建这个点的数据
                    index_ivf_->reconstruct(ids_in_cluster[idx], 
                                          centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim);
                    centroids_file.write((char*)&dim, sizeof(dim));
                    centroids_file.write((char*)(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim), dim * sizeof(float));
                }
            } else {
                // 如果cluster中的点数不足m，则重复使用已有点
                for (int j = 0; j < m_centroids_; j++) {
                    size_t idx = j % ids_in_cluster.size();
                    // 从IVF索引中重建这个点的数据
                    index_ivf_->reconstruct(ids_in_cluster[idx], 
                                          centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim);
                    centroids_file.write((char*)&dim, sizeof(dim));
                    centroids_file.write((char*)(centroids.data() + (i * (m_centroids_ + 1) + j + 1) * dim), dim * sizeof(float));
                }
            }
        }
        centroids_file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building centroids from IVF: " << e.what() << std::endl;
        return false;
    }
}

// 显式实例化模板类
template class IndexBuilder<float>;
template class IndexBuilder<unsigned char>;
template class IndexBuilder<int>;

} // namespace CNNS
