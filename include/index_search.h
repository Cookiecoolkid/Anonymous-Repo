#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <index_nsg.h>
#include <util.h>
#include "data_load.h"
#include "statistics.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <omp.h>
#include <tuple>

namespace CNNS {

enum DataLoadType {
    MMAP,
    PREAD,
};

// 搜索上下文结构体
struct SearchContext {
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, float*> cluster_data_map;
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;
    faiss::IndexHNSWFlat* index_hnsw;
    int n_clusters;
    int m;
    unsigned query_dim;
    int search_K;
    int search_L;
    unsigned k;
    std::string prefix;
};

// mmap版本的搜索上下文结构体
struct SearchContextMMap {
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, ClusterMMap> cluster_data_map;
    std::map<int, MappingMMap> id_mapping_map;
    faiss::IndexHNSWFlat* index_hnsw;
    int n_clusters;
    int m;
    unsigned query_dim;
    int search_K;
    int search_L;
    unsigned k;
    std::string prefix;
    std::atomic<unsigned> search_nhops{0};  // 搜索跳数统计
};

class IndexSearcher {
public:
    // 构造函数
    IndexSearcher(const std::string& prefix,
                 int search_K = 100,
                 int search_L = 100,
                 unsigned k = 100);

    // mmap版本的构造函数
    IndexSearcher(const std::string& prefix,
                 bool use_mmap,
                 int search_K = 100,
                 int search_L = 100,
                 unsigned k = 100);

    // 析构函数
    ~IndexSearcher();

    // 初始化搜索上下文
    bool initialize(const std::string& query_data_path,
                   const std::string& ground_truth_path);

    // 初始化mmap搜索上下文
    bool initialize_mmap(const std::string& query_data_path,
                        const std::string& ground_truth_path);

    // 执行搜索
    bool search(const float* query_data,
               unsigned query_num,
               const std::vector<std::vector<unsigned>>& ground_truth,
               int nprobe,
               std::vector<std::vector<unsigned>>& results,
               std::vector<double>& recalls,
               Statistics* stats = nullptr);

    bool search_mmap(const float* query_data,
                    unsigned query_num,
                    const std::vector<std::vector<unsigned>>& ground_truth,
                    int nprobe,
                    std::vector<std::vector<unsigned>>& results,
                    std::vector<double>& recalls,
                    Statistics* stats = nullptr,
                    int num_threads = -1);
    
    bool search_pread(const float* query_data,
               unsigned query_num,
               const std::vector<std::vector<unsigned>>& ground_truth,
               int nprobe,
               std::vector<std::vector<unsigned>>& results,
               std::vector<double>& recalls,
               Statistics* stats = nullptr);

    bool search_serial(const float* query_data,
               unsigned query_num,
               const std::vector<std::vector<unsigned>>& ground_truth,
               int nprobe,
               std::vector<std::vector<unsigned>>& results,
               std::vector<double>& recalls,
               Statistics* stats = nullptr);

    // 获取搜索统计信息
    void get_stats(double& total_time, double& recall_rate);

    // Getter函数
    unsigned get_query_dim() const { return ctx_.query_dim; }
    int get_search_K() const { return ctx_.search_K; }
    int get_search_L() const { return ctx_.search_L; }
    unsigned get_k() const { return ctx_.k; }

private:
    // 加载cluster数据
    bool load_cluster_data(int cluster_id,
                          unsigned global_dim,
                          float*& cluster_data,
                          unsigned& points_num);

    // 加载ID映射
    bool load_id_mapping(int cluster_id,
                        unsigned points_num,
                        std::vector<faiss::idx_t>& id_mapping);

    // 加载NSG索引
    bool load_nsg_index(int cluster_id,
                       unsigned dim,
                       unsigned points_num,
                       efanna2e::IndexNSG*& nsg_index);

    // 按需加载指定cluster的数据和NSG索引
    void load_cluster_specific_data_and_nsg(int cluster_id,
                                          unsigned global_dim);

    // 在HNSW图上搜索并获取排序后的cluster
    std::vector<std::pair<faiss::idx_t, int>> search_hnsw_and_sort_clusters(
        const float* query_data,
        int nprobe);

    // mmap版本的HNSW搜索并获取排序后的cluster
    std::vector<std::pair<faiss::idx_t, int>> search_hnsw_and_sort_clusters_mmap(
        const float* query_data,
        int nprobe,
        std::map<faiss::idx_t, faiss::idx_t>& cluster_nearest_point_ids);

    // 合并两个优先队列
    bool merge_topk_queue(
        std::priority_queue<std::pair<float, unsigned>>& target_queue,
        std::priority_queue<std::pair<float, unsigned>>& source_queue,
        unsigned k);

    // 在NSG上搜索单个cluster
    void search_nsg_cluster(
        faiss::idx_t cluster_id,
        const float* query_data,
        std::priority_queue<std::pair<float, unsigned>>& local_queue,
        float& current_min_max_dist,
        bool& query_early_stopped);

    // mmap版本的NSG搜索
    void search_nsg_cluster_mmap(
        faiss::idx_t cluster_id,
        const float* query_data,
        std::priority_queue<std::pair<float, unsigned>>& local_queue,
        float& current_min_max_dist,
        bool& query_early_stopped,
        SearchContextMMap& ctx_mmap,
        faiss::idx_t nearest_point_id);
    
    // mmap版本的NSG搜索（接受数据指针作为参数）
    void search_nsg_cluster_mmap_with_data(
        faiss::idx_t cluster_id,
        const float* query_data,
        std::priority_queue<std::pair<float, unsigned>>& local_queue,
        std::atomic<float>& current_max_min_dist,
        efanna2e::IndexNSG* nsg_index,
        const ClusterMMap& cluster_info,
        const MappingMMap& mapping_info,
        SearchContextMMap& ctx_mmap,
        faiss::idx_t nearest_point_id,
        std::atomic<unsigned>& local_search_nhops);
    
    void search_nsg_cluster_pread(
        faiss::idx_t cluster_id,
        const float* query_data,
        std::priority_queue<std::pair<float, unsigned>>& local_queue,
        float& current_min_max_dist,
        bool& query_early_stopped,
        SearchContextMMap& ctx_mmap);

    // 处理搜索结果
    std::vector<unsigned> process_search_results(
        std::priority_queue<std::pair<float, unsigned>>& topk_queue,
        unsigned k);

    // 计算单个查询的recall
    int calculate_query_recall(
        const std::vector<unsigned>& final_results,
        const std::unordered_set<unsigned>& ground_truth_set);

    // 计算最优线程分配策略
    std::tuple<int, int, int> calculate_optimal_thread_allocation(int total_threads, int query_num);

    // 成员变量
    SearchContext ctx_;
    SearchContextMMap ctx_mmap_;


    std::atomic<double> total_search_time_{0.0};
    std::atomic<int> total_correct_{0};
    std::atomic<int> total_ground_truth_{0};
};

} // namespace CNNS
