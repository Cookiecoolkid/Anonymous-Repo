#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <atomic>
#include <thread>
#include <algorithm>
#include <regex>
#include <fstream>
#include <filesystem>
#include "index_build.h"
#include "aux_util.h"

namespace CNNS {

struct ClusterInfo {
    faiss::idx_t cluster_id;
    std::string cluster_file;
    std::string nndescent_file;
    size_t points_count;
    unsigned dim;
};

// 从cluster文件名中提取cluster_id
faiss::idx_t extract_cluster_id(const std::string& filename) {
    std::regex pattern("cluster_(\\d+)\\.data");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        return std::stoull(match[1]);
    }
    return -1;
}

// 获取所有cluster文件信息
std::vector<ClusterInfo> get_cluster_files(const std::string& prefix) {
    std::vector<ClusterInfo> clusters;
    std::string cluster_dir = prefix + "/cluster_data";
    std::string nndescent_dir = prefix + "/nndescent";
    
    if (!std::filesystem::exists(cluster_dir)) {
        std::cerr << "Cluster directory does not exist: " << cluster_dir << std::endl;
        return clusters;
    }
    
    if (!std::filesystem::exists(nndescent_dir)) {
        std::cerr << "NNDescent directory does not exist: " << nndescent_dir << std::endl;
        return clusters;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(cluster_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".data") {
            std::string filename = entry.path().filename().string();
            faiss::idx_t cluster_id = extract_cluster_id(filename);
            
            if (cluster_id >= 0) {
                // 检查对应的NNDescent图是否存在
                std::string nndescent_file = nndescent_dir + "/nndescent_" + 
                                           std::to_string(cluster_id) + ".graph";
                
                if (std::filesystem::exists(nndescent_file)) {
                    size_t file_size = std::filesystem::file_size(entry.path());
                    // 假设每个点占用4字节（float）
                    size_t points_count = file_size / (4 * sizeof(float)); // 需要从其他地方获取dim
                    
                    clusters.push_back({
                        cluster_id,
                        entry.path().string(),
                        nndescent_file,
                        points_count,
                        0 // dim will be set later
                    });
                } else {
                    std::cout << "Warning: NNDescent graph not found for cluster " << cluster_id 
                              << ", skipping..." << std::endl;
                }
            }
        }
    }
    
    // 按cluster_id排序
    std::sort(clusters.begin(), clusters.end(), 
              [](const ClusterInfo& a, const ClusterInfo& b) {
                  return a.cluster_id < b.cluster_id;
              });
    
    return clusters;
}

// 从centroids文件获取维度信息
unsigned get_dimension_from_centroids(const std::string& prefix) {
    std::string centroids_file = prefix + "/centroids.data";
    std::ifstream file(centroids_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open centroids file: " << centroids_file << std::endl;
        return 0;
    }
    
    int n_clusters, m_centroids;
    unsigned dim;
    
    file.read((char*)&n_clusters, sizeof(n_clusters));
    file.read((char*)&m_centroids, sizeof(m_centroids));
    file.read((char*)&dim, sizeof(dim));
    
    file.close();
    return dim;
}

// 构建单个cluster的NSG图
bool build_nsg_for_cluster(faiss::idx_t cluster_id, 
                          const std::string& cluster_file,
                          const std::string& nndescent_file,
                          unsigned dim,
                          const std::string& prefix,
                          int L_nsg,
                          int R_nsg,
                          int C_nsg,
                          bool use_mmap) {
    try {
        std::cout << "[NSG] Building graph for cluster " << cluster_id 
                  << " from file: " << cluster_file << std::endl;
        
        // 加载cluster数据
        CNNS::ClusterMMap cluster_data;
        if (!load_cluster_data_mmap(cluster_id, dim, cluster_data, prefix)) {
            std::cerr << "Failed to load cluster data for cluster " << cluster_id << std::endl;
            return false;
        }
        
        if (!cluster_data.data_ptr) {
            std::cerr << "Invalid data pointer for cluster " << cluster_id << std::endl;
            return false;
        }
        
        // 获取实际点数
        size_t points_count = cluster_data.points_num;
        std::cout << "[NSG] Cluster " << cluster_id << " has " << points_count << " points" << std::endl;
        
        // 构建NSG
        efanna2e::IndexNSG index(dim, points_count, efanna2e::L2, nullptr);
        efanna2e::Parameters paras;
        paras.Set<unsigned>("L", L_nsg);
        paras.Set<unsigned>("R", R_nsg);
        paras.Set<unsigned>("C", C_nsg);
        paras.Set<std::string>("nn_graph_path", nndescent_file);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        index.Build(points_count, cluster_data.data_ptr, paras);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 保存NSG图
        std::string nsg_filename = prefix + "/nsg_graph/nsg_" + 
                                 std::to_string(cluster_id) + ".nsg";
        if (use_mmap) {
            index.Save_mmap_with_dist(nsg_filename.c_str());
        } else {
            index.Save(nsg_filename.c_str());
        }
        
        std::cout << "[NSG] Successfully built and saved graph for cluster " << cluster_id 
                  << " in " << duration.count() / 1000.0 << " s" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error building NSG for cluster " << cluster_id 
                  << ": " << e.what() << std::endl;
        return false;
    }
}

} // namespace CNNS

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <prefix> [L_nsg] [R_nsg] [C_nsg] [use_mmap]" << std::endl;
        std::cout << "Example: " << argv[0] << " /path/to/index 32 100 500 1" << std::endl;
        return 1;
    }
    
    std::string prefix = argv[1];
    
    // NSG参数（可选）
    int L_nsg = (argc > 2) ? std::stoi(argv[2]) : 32;
    int R_nsg = (argc > 3) ? std::stoi(argv[3]) : 100;
    int C_nsg = (argc > 4) ? std::stoi(argv[4]) : 500;
    bool use_mmap = (argc > 5) ? (std::stoi(argv[5]) != 0) : true;
    
    std::cout << "=== NSG Build Test ===" << std::endl;
    std::cout << "Prefix: " << prefix << std::endl;
    std::cout << "Parameters: L=" << L_nsg << ", R=" << R_nsg 
              << ", C=" << C_nsg << ", use_mmap=" << use_mmap << std::endl;
    
    // 确保nsg_graph目录存在
    std::string nsg_dir = prefix + "/nsg_graph";
    std::filesystem::create_directories(nsg_dir);
    
    // 获取维度信息
    unsigned dim = CNNS::get_dimension_from_centroids(prefix);
    if (dim == 0) {
        std::cerr << "Failed to get dimension from centroids file" << std::endl;
        return 1;
    }
    std::cout << "Dimension: " << dim << std::endl;
    
    // 获取所有cluster文件
    auto clusters = CNNS::get_cluster_files(prefix);
    if (clusters.empty()) {
        std::cerr << "No cluster files found in " << prefix << "/cluster_data" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << clusters.size() << " cluster files with NNDescent graphs" << std::endl;
    
    // 设置并发参数
    // const int max_parallel_clusters = 64;  // NSG阶段使用更少的并发
    std::atomic<int> running_clusters(0);
    std::atomic<int> completed_clusters(0);
    std::atomic<int> failed_clusters(0);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 并行构建NSG图
    // #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        // 控制并发数量
        /*
        while (running_clusters.load() >= max_parallel_clusters) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        */
        running_clusters.fetch_add(1);
        
        const auto& cluster = clusters[i];
        
        bool success = CNNS::build_nsg_for_cluster(
            cluster.cluster_id, cluster.cluster_file, cluster.nndescent_file, 
            dim, prefix, L_nsg, R_nsg, C_nsg, use_mmap
        );
        
        if (success) {
            completed_clusters.fetch_add(1);
        } else {
            failed_clusters.fetch_add(1);
        }
        
        running_clusters.fetch_sub(1);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\n=== NSG Build Summary ===" << std::endl;
    std::cout << "Total clusters: " << clusters.size() << std::endl;
    std::cout << "Completed: " << completed_clusters.load() << std::endl;
    std::cout << "Failed: " << failed_clusters.load() << std::endl;
    std::cout << "Total time: " << total_duration.count() / 1000.0 << " s" << std::endl;
    
    if (completed_clusters.load() > 0) {
        std::cout << "Successfully built NSG graphs for " << completed_clusters.load() 
                  << " clusters" << std::endl;
        return 0;
    } else {
        std::cout << "Failed to build any NSG graphs" << std::endl;
        return 1;
    }
} 