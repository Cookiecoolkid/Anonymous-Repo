#include "index_search.h"
#include "aux_util.h"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace CNNS {

IndexSearcher::IndexSearcher(const std::string& prefix,
                           int search_K,
                           int search_L,
                           unsigned k)
    : ctx_() {
    ctx_.prefix = prefix;
    ctx_.search_K = search_K;
    ctx_.search_L = search_L;
    ctx_.k = k;
}

IndexSearcher::~IndexSearcher() {
    // 清理资源
    delete ctx_.index_hnsw;
    for (auto& pair : ctx_.cluster_nsg_indices) {
        delete pair.second;
    }
    for (auto& pair : ctx_.cluster_data_map) {
        delete[] pair.second;
    }
}

bool IndexSearcher::initialize(const std::string& query_data_path,
                             const std::string& ground_truth_path) {
    try {
        // 加载查询数据
        unsigned query_num;
        std::vector<float> query_data = CNNS::load_fvecs(query_data_path, query_num, ctx_.query_dim);
        std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(ground_truth_path.c_str());

        // 加载质心数据
        unsigned centroids_dim;
        std::vector<float> centroids = CNNS::load_centroids(ctx_.prefix + "/centroids.data", 
                                                          ctx_.n_clusters, ctx_.m, centroids_dim);
        if (centroids_dim != ctx_.query_dim) {
            throw std::runtime_error("Dimension mismatch between data and centroids");
        }

        // 加载HNSW图索引
        ctx_.index_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(
            faiss::read_index((ctx_.prefix + "/hnsw_memory.index").c_str()));
        if (!ctx_.index_hnsw) {
            throw std::runtime_error("Error loading HNSW index from " + ctx_.prefix + "/hnsw_memory.index");
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing searcher: " << e.what() << std::endl;
        return false;
    }
}

bool IndexSearcher::search(const float* query_data,
                          unsigned query_num,
                          const std::vector<std::vector<unsigned>>& ground_truth,
                          int nprobe,
                          std::vector<std::vector<unsigned>>& results,
                          std::vector<double>& recalls,
                          Statistics* stats) {
    try {
        results.resize(query_num);
        recalls.resize(query_num);
        total_correct_ = 0;
        total_ground_truth_ = 0;
        total_search_time_ = 0.0;

        auto start_time_search = std::chrono::high_resolution_clock::now();
        std::vector<double> query_times; // 记录每个查询的时间
        query_times.reserve(query_num);

        // 设置 omp 线程数
        omp_set_num_threads(8);

        // 并行处理每个查询
        // #pragma omp parallel
        {
            int local_correct = 0;
            int local_total = 0;
            std::mutex queue_mutex;

            // #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < query_num; i++) {
                auto query_start = std::chrono::high_resolution_clock::now();
                
                // 在HNSW图上搜索并获取排序后的cluster
                auto sorted_clusters = search_hnsw_and_sort_clusters(query_data + i * ctx_.query_dim, nprobe);

                std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
                std::priority_queue<std::pair<float, unsigned>> topk_queue;
                float current_min_max_dist = std::numeric_limits<float>::max();
                bool query_early_stopped = false;

                // 使用OpenMP parallel for并行处理clusters
                #pragma omp parallel for schedule(dynamic)
                for (size_t cluster_idx = 0; cluster_idx < sorted_clusters.size(); ++cluster_idx) {
                    if (query_early_stopped) continue;

                    faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;
                    
                    // 加载cluster数据
                    bool success_load = false;
                    {
                        #pragma omp critical(cluster_load)
                        {
                            if (!ctx_.cluster_nsg_indices.count(cluster_id)) {
                                load_cluster_specific_data_and_nsg(cluster_id, ctx_.query_dim);
                            } else {
                                success_load = true;
                            }
                        }
                    }

                    if (success_load && !query_early_stopped) {
                        std::priority_queue<std::pair<float, unsigned>> local_queue;
                        float local_max_dist = current_min_max_dist;

                        search_nsg_cluster(cluster_id, 
                                         query_data + i * ctx_.query_dim,
                                         local_queue, local_max_dist, 
                                         query_early_stopped);

                        // 合并结果
                        {
                            std::lock_guard<std::mutex> guard(queue_mutex);
                            merge_topk_queue(topk_queue, local_queue, ctx_.k);
                            current_min_max_dist = topk_queue.top().first;
                        }
                    }
                }

                // 处理搜索结果
                results[i] = process_search_results(topk_queue, ctx_.k);
                
                // 计算recall
                int query_correct_count = calculate_query_recall(results[i], ground_truth_set);
                recalls[i] = static_cast<double>(query_correct_count) / ground_truth_set.size();
                local_correct += query_correct_count;
                local_total += ground_truth_set.size();
                
                // 记录单个查询时间
                auto query_end = std::chrono::high_resolution_clock::now();
                double query_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                    query_end - query_start).count();
                query_times.push_back(query_time);
            }

            // 合并线程本地统计信息
            #pragma omp critical(update_stats)
            {
                total_correct_ += local_correct;
                total_ground_truth_ += local_total;
            }
        }

        auto end_time_search = std::chrono::high_resolution_clock::now();
        total_search_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time_search - start_time_search).count();

        // 记录统计信息到Statistics对象
        if (stats) {
            stats->record_search_time(total_search_time_);
            stats->record_query_results(recalls);
            stats->record_qps(query_num, total_search_time_);
            stats->record_latency_percentiles(query_times);
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during search: " << e.what() << std::endl;
        return false;
    }
}

void IndexSearcher::get_stats(double& total_time, double& recall_rate) {
    total_time = total_search_time_;
    recall_rate = static_cast<double>(total_correct_) / total_ground_truth_;
}

std::vector<std::pair<faiss::idx_t, int>> IndexSearcher::search_hnsw_and_sort_clusters(
    const float* query_data,
    int nprobe) {
    
    std::vector<float> query_distances(nprobe);
    std::vector<faiss::idx_t> query_labels(nprobe);
    
    ctx_.index_hnsw->search(1, query_data, nprobe, query_distances.data(), query_labels.data());

    // 统计每个cluster包含的样本点数量
    std::map<faiss::idx_t, int> cluster_sample_count;
    for (int j = 0; j < nprobe; ++j) {
        faiss::idx_t point_id = query_labels[j];
        faiss::idx_t cluster_id = point_id / (ctx_.m + 1);
        cluster_sample_count[cluster_id]++;
    }

    // 将cluster按样本点数量排序
    std::vector<std::pair<faiss::idx_t, int>> sorted_clusters;
    for (const auto& pair : cluster_sample_count) {
        sorted_clusters.push_back(pair);
    }
    std::sort(sorted_clusters.begin(), sorted_clusters.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });

    return sorted_clusters;
}

bool IndexSearcher::merge_topk_queue(
    std::priority_queue<std::pair<float, unsigned>>& target_queue,
    std::priority_queue<std::pair<float, unsigned>>& source_queue,
    unsigned k) {
    
    bool has_update = false;
    // 将source_queue中的所有元素添加到target_queue
    while (!source_queue.empty()) {
        auto [dist, id] = source_queue.top();
        source_queue.pop();
        
        if (target_queue.size() < k) {
            target_queue.push({dist, id});
            has_update = true;
        } else if (dist < target_queue.top().first) {
            target_queue.pop();
            target_queue.push({dist, id});
            has_update = true;
        }
    }
    return has_update;
}

void IndexSearcher::search_nsg_cluster(
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& local_queue,
    float& current_min_max_dist,
    bool& query_early_stopped) {

    efanna2e::IndexNSG* nsg_index = ctx_.cluster_nsg_indices.at(cluster_id);
    float* current_cluster_data = ctx_.cluster_data_map.at(cluster_id);
    const auto& current_id_mapping = ctx_.id_mapping_map.at(cluster_id);
    
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", ctx_.search_L);
    paras.Set<unsigned>("P_search", ctx_.search_L);
    paras.Set<unsigned>("K_search", ctx_.search_K);
    std::vector<unsigned> tmp(ctx_.search_K);
    nsg_index->Search(query_data, current_cluster_data, ctx_.search_K, paras, tmp.data());

    float cluster_min_dist = std::numeric_limits<float>::max();
    std::unordered_map<unsigned, float> local_results;

    // 计算当前cluster中所有点的距离
    for (int m_loop = 0; m_loop < ctx_.search_K; m_loop++) {
        unsigned local_id = tmp[m_loop];
        if (local_id >= current_id_mapping.size()) continue;
        unsigned global_id = current_id_mapping[local_id];
        float dist = 0;
        for (unsigned d_idx = 0; d_idx < ctx_.query_dim; d_idx++) {
            float diff = query_data[d_idx] - current_cluster_data[local_id * ctx_.query_dim + d_idx];
            dist += diff * diff;
        }
        cluster_min_dist = std::min(cluster_min_dist, dist);
        local_results[global_id] = dist;
    }

    // 更新本地优先队列
    for (const auto& [global_id, dist] : local_results) {
        if (local_queue.size() < ctx_.k) {
            local_queue.push({dist, global_id});
        } else if (dist < local_queue.top().first) {
            local_queue.pop();
            local_queue.push({dist, global_id});
        }
    }

    // 如果当前cluster的最小距离大于等于当前topk中的最大距离，且队列已满，则提前停止
    if (cluster_min_dist >= current_min_max_dist && local_queue.size() >= ctx_.k) {
        query_early_stopped = true;
    }
}

std::vector<unsigned> IndexSearcher::process_search_results(
    std::priority_queue<std::pair<float, unsigned>>& topk_queue,
    unsigned k) {
    
    std::vector<unsigned> final_results;
    final_results.reserve(k);
    
    // 将优先队列中的结果按距离从小到大排序
    std::vector<std::pair<float, unsigned>> sorted_results;
    sorted_results.reserve(topk_queue.size());
    while (!topk_queue.empty()) {
        sorted_results.push_back(topk_queue.top());
        topk_queue.pop();
    }
    std::sort(sorted_results.begin(), sorted_results.end());
    
    // 提取前k个结果
    for (unsigned i = 0; i < k && i < sorted_results.size(); i++) {
        final_results.push_back(sorted_results[i].second);
    }
    
    return final_results;
}

int IndexSearcher::calculate_query_recall(
    const std::vector<unsigned>& final_results,
    const std::unordered_set<unsigned>& ground_truth_set) {
    
    int correct = 0;
    for (unsigned id : final_results) {
        if (ground_truth_set.count(id)) {
            correct++;
        }
    }
    return correct;
}


// =============================== LOAD DATA ===============================
bool IndexSearcher::load_cluster_data(int cluster_id,
                                    unsigned global_dim,
                                    float*& cluster_data,
                                    unsigned& points_num) {
    std::string cluster_filename = ctx_.prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".data";
    std::ifstream in_cluster_data(cluster_filename, std::ios::binary);
    if (!in_cluster_data.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open cluster file " << cluster_filename << std::endl;
        return false;
    }

    // 获取文件大小
    in_cluster_data.seekg(0, std::ios::end);
    size_t fsize = in_cluster_data.tellg();
    in_cluster_data.seekg(0, std::ios::beg);

    // 计算点数
    points_num = fsize / (global_dim * sizeof(float));
    if (points_num == 0) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cluster " << cluster_id << " has 0 points in " << cluster_filename << std::endl;
        in_cluster_data.close();
        return false;
    }

    // 分配内存
    cluster_data = new (std::nothrow) float[points_num * global_dim];
    if (!cluster_data) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for cluster_data for cluster " << cluster_id << std::endl;
        in_cluster_data.close();
        return false;
    }

    // 直接读取整个数据块
    in_cluster_data.read((char*)cluster_data, points_num * global_dim * sizeof(float));
    if ((unsigned)in_cluster_data.gcount() != points_num * global_dim * sizeof(float)) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error reading cluster data from " << cluster_filename << std::endl;
        delete[] cluster_data;
        cluster_data = nullptr;
        in_cluster_data.close();
        return false;
    }

    in_cluster_data.close();
    return true;
}

bool IndexSearcher::load_id_mapping(int cluster_id,
                                  unsigned points_num,
                                  std::vector<faiss::idx_t>& id_mapping) {
    std::string mapping_filename = ctx_.prefix + "/mapping/mapping_" + std::to_string(cluster_id);
    std::ifstream mapping_file(mapping_filename, std::ios::binary);
    if (!mapping_file.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open mapping file " << mapping_filename << std::endl;
        return false;
    }

    id_mapping.resize(points_num);
    mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
    if ((unsigned)mapping_file.gcount() != points_num * sizeof(faiss::idx_t)) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error reading id_mapping file " << mapping_filename << std::endl;
        mapping_file.close();
        return false;
    }
    mapping_file.close();
    return true;
}

bool IndexSearcher::load_nsg_index(int cluster_id,
                                 unsigned dim,
                                 unsigned points_num,
                                 efanna2e::IndexNSG*& nsg_index) {
    nsg_index = new (std::nothrow) efanna2e::IndexNSG(dim, points_num, efanna2e::L2, nullptr);
    if (!nsg_index) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for nsg_index for cluster " << cluster_id << std::endl;
        return false;
    }

    std::string nsg_filename = ctx_.prefix + "/nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
    try {
        nsg_index->Load(nsg_filename.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error loading NSG from " << nsg_filename << ": " << e.what() << std::endl;
        delete nsg_index;
        nsg_index = nullptr;
        return false;
    }
    return true;
}

void IndexSearcher::load_cluster_specific_data_and_nsg(int cluster_id,
                                                     unsigned global_dim) {
    // 加载cluster数据
    float* cluster_data = nullptr;
    unsigned points_num = 0;
    if (!load_cluster_data(cluster_id, global_dim, cluster_data, points_num)) {
        return;
    }

    // 加载ID映射
    std::vector<faiss::idx_t> id_mapping;
    if (!load_id_mapping(cluster_id, points_num, id_mapping)) {
        delete[] cluster_data;
        return;
    }

    // 加载NSG索引
    efanna2e::IndexNSG* nsg_index = nullptr;
    if (!load_nsg_index(cluster_id, global_dim, points_num, nsg_index)) {
        delete[] cluster_data;
        return;
    }

    // 成功加载，存入map
    ctx_.cluster_data_map[cluster_id] = cluster_data;
    ctx_.id_mapping_map[cluster_id] = id_mapping;
    ctx_.cluster_nsg_indices[cluster_id] = nsg_index;
}


} // namespace CNNS
