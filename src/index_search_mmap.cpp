#include "index_search.h"
#include "aux_util.h"
#include "data_load.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <atomic>

namespace CNNS {

IndexSearcher::IndexSearcher(const std::string& prefix,
                           bool use_mmap,
                           int search_K,
                           int search_L,
                           unsigned k)
    : ctx_() {
    ctx_.prefix = prefix;
    ctx_.search_K = search_K;
    ctx_.search_L = search_L;
    ctx_.k = k;

    // 初始化mmap相关的上下文
    ctx_mmap_.prefix = prefix;
    ctx_mmap_.search_K = search_K;
    ctx_mmap_.search_L = search_L;
    ctx_mmap_.k = k;
    ctx_mmap_.query_dim = 0;  // 将在initialize时设置
    ctx_mmap_.n_clusters = 0; // 将在initialize时设置
    ctx_mmap_.m = 0;         // 将在initialize时设置
    ctx_mmap_.index_hnsw = nullptr;
}

bool IndexSearcher::initialize_mmap(const std::string& query_data_path,
                                  const std::string& ground_truth_path) {
    try {
        // 加载查询数据
        unsigned query_num;
        std::vector<float> query_data = CNNS::load_fvecs(query_data_path, query_num, ctx_mmap_.query_dim);
        std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(ground_truth_path.c_str());

        // 加载质心数据
        unsigned centroids_dim;
        std::vector<float> centroids = CNNS::load_centroids(ctx_mmap_.prefix + "/centroids.data", 
                                                          ctx_mmap_.n_clusters, ctx_mmap_.m, centroids_dim);
        if (centroids_dim != ctx_mmap_.query_dim) {
            throw std::runtime_error("Dimension mismatch between data and centroids");
        }

        size_t init_rss = getCurrentRSS();

        // 加载HNSW图索引
        ctx_mmap_.index_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(
            faiss::read_index((ctx_mmap_.prefix + "/hnsw_memory.index").c_str()));
        if (!ctx_mmap_.index_hnsw) {
            throw std::runtime_error("Error loading HNSW index from " + ctx_mmap_.prefix + "/hnsw_memory.index");
        }

        // 打印当前内存使用情况
        std::cout << "Current memory usage for Graph Abstraction: " << getCurrentRSS() - init_rss << " MB" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing mmap searcher: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::pair<faiss::idx_t, int>> IndexSearcher::search_hnsw_and_sort_clusters_mmap(
    const float* query_data,
    int nprobe,
    std::map<faiss::idx_t, faiss::idx_t>& cluster_nearest_point_ids) {
    
    std::vector<float> query_distances(nprobe);
    std::vector<faiss::idx_t> query_labels(nprobe);
    
    ctx_mmap_.index_hnsw->search(1, query_data, nprobe, query_distances.data(), query_labels.data());

    /*
    std::cout << "Graph Abstraction Search Stats: " << faiss::hnsw_stats.n1 << ", " 
                    << faiss::hnsw_stats.n2 << ", " << faiss::hnsw_stats.ndis << ", " 
                    << faiss::hnsw_stats.nhops << std::endl;
    */
    // 统计每个cluster包含的样本点数量，并记录每个cluster中最近的point_id
    std::map<faiss::idx_t, int> cluster_sample_count;
    cluster_nearest_point_ids.clear(); // 清空之前的记录

    for (int j = 0; j < nprobe; ++j) {
        faiss::idx_t point_id = query_labels[j];
        faiss::idx_t cluster_id = point_id / (ctx_mmap_.m + 1);
        float distance = query_distances[j];
        
        cluster_sample_count[cluster_id]++;
        
        // 更新该cluster中最近的point_id
        if (cluster_nearest_point_ids.find(cluster_id) == cluster_nearest_point_ids.end()) {
            cluster_nearest_point_ids[cluster_id] = point_id;
        } else {
            // 如果当前距离更小，更新为最近的point_id
            faiss::idx_t current_nearest = cluster_nearest_point_ids[cluster_id];
            float current_distance = std::numeric_limits<float>::max();
            
            // 找到当前nearest point对应的距离
            for (int k = 0; k < nprobe; ++k) {
                if (query_labels[k] == current_nearest) {
                    current_distance = query_distances[k];
                    break;
                }
            }
            
            if (distance < current_distance) {
                cluster_nearest_point_ids[cluster_id] = point_id;
            }
        }
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


void IndexSearcher::search_nsg_cluster_mmap(
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& local_queue,
    float& current_min_max_dist,
    bool& query_early_stopped,
    SearchContextMMap& ctx_mmap,
    faiss::idx_t nearest_point_id) {

    efanna2e::IndexNSG* nsg_index = ctx_mmap.cluster_nsg_indices.at(cluster_id);
    const ClusterMMap& cluster_info = ctx_mmap.cluster_data_map.at(cluster_id);
    const MappingMMap& mapping_info = ctx_mmap.id_mapping_map.at(cluster_id);
    
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", ctx_mmap.search_L);
    paras.Set<unsigned>("P_search", ctx_mmap.search_L);
    paras.Set<unsigned>("K_search", ctx_mmap.search_K);
    std::vector<unsigned> tmp(ctx_mmap.search_K);
    std::vector<float> distances(ctx_mmap.search_K);

    // global to local
    unsigned internal_nearest_point_id = nearest_point_id % (ctx_mmap.m + 1);

    // 使用Search_mmap_with_dist进行搜索，它会同时返回距离
    nsg_index->Search_mmap_with_dist(query_data, cluster_info.data_ptr, ctx_mmap.search_K, paras, tmp.data(), distances.data(), internal_nearest_point_id);

    float cluster_min_dist = std::numeric_limits<float>::max();
    std::unordered_map<unsigned, float> local_results;

    // 使用从Search_mmap_with_dist返回的距离
    for (int m_loop = 0; m_loop < ctx_mmap.search_K; m_loop++) {
        unsigned local_id = tmp[m_loop];
        if (local_id >= mapping_info.length) continue;
        unsigned global_id = mapping_info.data[local_id];
        float dist = distances[m_loop];
        cluster_min_dist = std::min(cluster_min_dist, dist);
        local_results[global_id] = dist;
    }

    // 更新本地优先队列
    for (const auto& [global_id, dist] : local_results) {
        if (local_queue.size() < ctx_mmap.k) {
            local_queue.push({dist, global_id});
        } else if (dist < local_queue.top().first) {
            local_queue.pop();
            local_queue.push({dist, global_id});
        }
    }

    // 如果当前cluster的最小距离大于等于当前topk中的最大距离，且队列已满，则提前停止
    if (cluster_min_dist >= current_min_max_dist && local_queue.size() == ctx_mmap.k) {
        query_early_stopped = true; 
    }
}

void IndexSearcher::search_nsg_cluster_mmap_with_data(
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& local_queue,
    std::atomic<float>& current_max_min_dist,
    efanna2e::IndexNSG* nsg_index,
    const ClusterMMap& cluster_info,
    const MappingMMap& mapping_info,
    SearchContextMMap& ctx_mmap,
    faiss::idx_t nearest_point_id,
    std::atomic<unsigned>& local_search_nhops) {

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", ctx_mmap.search_L);
    paras.Set<unsigned>("P_search", ctx_mmap.search_L);
    paras.Set<unsigned>("K_search", ctx_mmap.search_K);
    std::vector<unsigned> tmp(ctx_mmap.search_K);
    std::vector<float> distances(ctx_mmap.search_K);

    // global to local
    unsigned internal_nearest_point_id = nearest_point_id % (ctx_mmap.m + 1);

    // 统计距离计算次数
    unsigned distance_computations = 0;

    // 使用Search_mmap_with_dist进行搜索，它会同时返回距离
    nsg_index->Search_mmap_with_dist(query_data, cluster_info.data_ptr, ctx_mmap.search_K, paras, tmp.data(), distances.data(), internal_nearest_point_id, &distance_computations);

    // 累加到全局统计
    local_search_nhops.fetch_add(distance_computations, std::memory_order_relaxed);

    float cluster_min_dist = std::numeric_limits<float>::max();
    std::unordered_map<unsigned, float> local_results;

    // 使用从Search_mmap_with_dist返回的距离
    for (int m_loop = 0; m_loop < ctx_mmap.search_K; m_loop++) {
        unsigned local_id = tmp[m_loop];
        if (local_id >= mapping_info.length) continue;
        unsigned global_id = mapping_info.data[local_id];
        float dist = distances[m_loop];
        cluster_min_dist = std::min(cluster_min_dist, dist);
        local_results[global_id] = dist;
    }

    // 更新本地优先队列
    for (const auto& [global_id, dist] : local_results) {
        if (local_queue.size() < ctx_mmap.k) {
            local_queue.push({dist, global_id});
        } else if (dist < local_queue.top().first) {
            local_queue.pop();
            local_queue.push({dist, global_id});
        }
    }

    // 更新当前最大最小距离
    // float current_value = current_max_min_dist.load(std::memory_order_relaxed);
    // float new_value = std::max(current_value, cluster_min_dist);
    // current_max_min_dist.store(new_value, std::memory_order_relaxed);
}

bool IndexSearcher::search_mmap(const float* query_data,
                          unsigned query_num,
                          const std::vector<std::vector<unsigned>>& ground_truth,
                          int nprobe,
                          std::vector<std::vector<unsigned>>& results,
                          std::vector<double>& recalls,
                          Statistics* stats,
                          int num_threads) {
    try {
        results.resize(query_num);
        recalls.resize(query_num);
        total_correct_ = 0;
        total_ground_truth_ = 0;
        total_search_time_ = 0.0;

        // 重置本地图搜索跳数统计
        ctx_mmap_.search_nhops.store(0, std::memory_order_relaxed);

        auto start_time_search = std::chrono::high_resolution_clock::now();
        std::vector<double> query_times; // 记录每个查询的时间
        query_times.reserve(query_num);

        // 获取系统总线程数并计算最优分配
        int system_threads = omp_get_max_threads();
        int total_threads;
        if (num_threads == -1) {
            total_threads = system_threads;  // 自动检测
        } else {
            total_threads = std::min(num_threads, system_threads);  // 取较小值
        }
        auto [query_threads, cluster_threads, batch_size] = calculate_optimal_thread_allocation(total_threads, query_num);
        
        std::cout << "Thread allocation - System: " << system_threads 
                  << ", Requested: " << (num_threads == -1 ? "auto" : std::to_string(num_threads))
                  << ", Used: " << total_threads
                  << ", Query threads: " << query_threads 
                  << ", Cluster threads: " << cluster_threads 
                  << ", Batch size: " << batch_size << std::endl;

        // 使用嵌套并行：外层并行处理Query，内层并行处理Cluster
        #pragma omp parallel num_threads(query_threads)
        {
            // 线程本地变量
            int local_correct = 0;
            int local_total = 0;
            std::vector<double> local_query_times;
            
            // 并行处理Query
            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < query_num; i++) {
                auto query_start = std::chrono::high_resolution_clock::now();
                
                // 在HNSW图上搜索并获取排序后的cluster
                std::map<faiss::idx_t, faiss::idx_t> cluster_nearest_point_ids;
                std::vector<std::pair<faiss::idx_t, int>> sorted_clusters = search_hnsw_and_sort_clusters_mmap(query_data + i * ctx_mmap_.query_dim, nprobe, cluster_nearest_point_ids);

                std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
                std::priority_queue<std::pair<float, unsigned>> topk_queue;
                std::atomic<float> current_max_min_dist(std::numeric_limits<float>::max());
                std::atomic<bool> query_early_stopped(false);
                std::atomic<int> consecutive_no_contribution(0); // 连续没有贡献的batch数量
                const int EARLY_STOP_THRESHOLD = sorted_clusters.size() / 4;
                // const int EARLY_STOP_THRESHOLD = 20;
                for (size_t batch_start = 0; batch_start < sorted_clusters.size(); batch_start += batch_size) {

                    size_t batch_end = std::min(batch_start + batch_size, sorted_clusters.size());
                    if (query_early_stopped.load(std::memory_order_relaxed)) {
                        // std::cout << "Ahhhh Break!!! Cluster Num: " << sorted_clusters.size() 
                        //           << " And Current Batch: " << batch_start << " to " << batch_end << std::endl;
                        break;
                    }
                    // 使用线程本地存储收集batch结果
                    std::vector<std::priority_queue<std::pair<float, unsigned>>> thread_local_queues(batch_end - batch_start);

                    // 内层并行：使用剩余线程处理cluster
                    #pragma omp parallel for num_threads(cluster_threads) schedule(dynamic, 1)
                    for (size_t cluster_offset = 0; cluster_offset < (batch_end - batch_start); ++cluster_offset) {
                        size_t cluster_idx = batch_start + cluster_offset;

                        faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;

                        // cluster load: critical section
                        {
                            efanna2e::IndexNSG* nsg_ptr = nullptr;
                            const ClusterMMap* cluster_ptr = nullptr;
                            const MappingMMap* mapping_ptr = nullptr;

                            #pragma omp critical(cluster_load)
                            {
                                if (!ctx_mmap_.cluster_nsg_indices.count(cluster_id)) {
                                    load_cluster_specific_data_and_nsg_mmap(cluster_id, ctx_mmap_.query_dim, 
                                        ctx_mmap_.cluster_data_map, ctx_mmap_.id_mapping_map, 
                                        ctx_mmap_.cluster_nsg_indices, ctx_mmap_.prefix);
                                }
                                
                                // 在临界区内获取数据指针，避免竞态条件
                                nsg_ptr = ctx_mmap_.cluster_nsg_indices.at(cluster_id);
                                cluster_ptr = &ctx_mmap_.cluster_data_map.at(cluster_id);
                                mapping_ptr = &ctx_mmap_.id_mapping_map.at(cluster_id);
                            }
                            if (nsg_ptr != nullptr && 
                                cluster_ptr != nullptr && 
                                mapping_ptr != nullptr) {
                                
                                // 执行cluster搜索，直接传递数据指针
                                faiss::idx_t global_nearest_point_id = cluster_nearest_point_ids.count(cluster_id) ? 
                                      cluster_nearest_point_ids[cluster_id] : 0;
                                
                                search_nsg_cluster_mmap_with_data(cluster_id, query_data + i * ctx_mmap_.query_dim,
                                                                  thread_local_queues[cluster_offset], current_max_min_dist, 
                                                                  nsg_ptr, *cluster_ptr, *mapping_ptr, ctx_mmap_, global_nearest_point_id, ctx_mmap_.search_nhops);
                            }
                            
                        }
                    }

                    // 批量合并所有线程本地队列到全局队列，并检查是否有贡献
                    bool batch_has_contribution = false;
                    for (auto& local_queue : thread_local_queues) {
                        if (!local_queue.empty()) {
                            bool has_update = merge_topk_queue(topk_queue, local_queue, ctx_mmap_.k);
                            if (has_update) {
                                batch_has_contribution = true;
                            }
                        }
                    }
                    
                    // 更新当前最大最小距离
                    if (topk_queue.size() == ctx_mmap_.k) {
                        current_max_min_dist.store(topk_queue.top().first, std::memory_order_relaxed);
                    }
                    
                    // 检查early stop条件：连续没有贡献的batch数量
                    if (!batch_has_contribution) {
                        int current_count = consecutive_no_contribution.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (current_count >= EARLY_STOP_THRESHOLD) {
                            query_early_stopped.store(true, std::memory_order_relaxed);
                            // std::cout << "Early Stop = True !!! Cluster Num: " << sorted_clusters.size() 
                            //           << " And Current Batch: " << batch_start << " to " << batch_end << std::endl;
                            break;
                        }
                    } else {
                        // 如果当前batch有贡献，重置计数器
                        consecutive_no_contribution.store(0, std::memory_order_relaxed);
                    }

                }

                // 处理搜索结果
                results[i] = process_search_results(topk_queue, ctx_mmap_.k);
                
                // 计算recall
                int query_correct_count = calculate_query_recall(results[i], ground_truth_set);
                recalls[i] = static_cast<double>(query_correct_count) / ground_truth_set.size();
                local_correct += query_correct_count;
                local_total += ground_truth_set.size();
                
                // 记录单个查询时间
                auto query_end = std::chrono::high_resolution_clock::now();
                double query_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                    query_end - query_start).count();
                local_query_times.push_back(query_time);
            }
            
            // 线程本地结果合并到全局结果
            #pragma omp critical(result_merge)
            {
                total_correct_ += local_correct;
                total_ground_truth_ += local_total;
                query_times.insert(query_times.end(), local_query_times.begin(), local_query_times.end());
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
            stats->record_graph_abstraction_nhops(faiss::hnsw_stats.nhops);
            stats->record_local_graph_nhops(ctx_mmap_.search_nhops.load(std::memory_order_relaxed));
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during search: " << e.what() << std::endl;
        return false;
    }
}

// 计算最优线程分配策略
std::tuple<int, int, int> IndexSearcher::calculate_optimal_thread_allocation(int total_threads, int query_num) {
    int query_threads, cluster_threads, batch_size;
    
    // 首先确定batch_size策略，但不超过总cluster数量的10%
    int max_batch_size = std::max(1, ctx_mmap_.n_clusters / 10); // 不超过cluster数量的10%，但至少为1
    
    if (total_threads <= 4) {
        batch_size = std::min(1, max_batch_size);
    } else if (total_threads <= 24) {
        batch_size = std::min(2, max_batch_size);
    } else {
        batch_size = std::min(4, max_batch_size);
    } 
    
    // 根据总线程数分配query和cluster线程
    if (total_threads <= 4) {
        // 线程数较少时，全部给query_threads
        query_threads = total_threads;
        cluster_threads = 0;
    } else if (total_threads <= 24) {
        // 中等线程数，大部分给query_threads
        query_threads = total_threads * 4 / 5;  // 4/5给query
        cluster_threads = total_threads - query_threads;
    } else {
        // 大量线程数，大部分分配给query
        query_threads = total_threads * 7 / 8;  // 7/8给query
        cluster_threads = total_threads - query_threads;
    }
    
    // 确保最小分配
    query_threads = std::max(1, query_threads);
    cluster_threads = std::max(0, cluster_threads);
    
    // 根据query数量调整
    if (query_num < query_threads) {
        query_threads = query_num;
        cluster_threads = total_threads - query_threads;
        
        // 重新检查batch_size是否合适
        if (cluster_threads > batch_size) {
            if (cluster_threads <= 1) {
                batch_size = std::min(1, max_batch_size);
            } else if (cluster_threads <= 2) {
                batch_size = std::min(2, max_batch_size);
            } else {
                batch_size = std::min(4, max_batch_size);
            }
            
            if (batch_size < cluster_threads) {
                cluster_threads = batch_size;
            }
        }
    }

    query_threads = total_threads;
    cluster_threads = 0;
    batch_size = total_threads >= 24 ? 4 : 2;
    // batch_size = 1;
    return {query_threads, cluster_threads, batch_size};
}

/*
// Old version 1 (no batch)
bool IndexSearcher::search_mmap(const float* query_data,
                          unsigned query_num,
                          const std::vector<std::vector<unsigned>>& ground_truth,
                          int nprobe,
                          std::vector<std::vector<unsigned>>& results,
                          std::vector<double>& recalls) {
    try {
        results.resize(query_num);
        recalls.resize(query_num);
        total_correct_ = 0;
        total_ground_truth_ = 0;
        total_search_time_ = 0.0;

        auto start_time_search = std::chrono::high_resolution_clock::now();

        // 设置 omp 线程数
        omp_set_num_threads(8);

        {
            int local_correct = 0;
            int local_total = 0;
            std::mutex queue_mutex;

            // #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < query_num; i++) {
                // 在HNSW图上搜索并获取排序后的cluster
                auto sorted_clusters = search_hnsw_and_sort_clusters_mmap(query_data + i * ctx_mmap_.query_dim, nprobe);

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
                    {
                        #pragma omp critical(cluster_load)
                        {
                            if (!ctx_mmap_.cluster_nsg_indices.count(cluster_id)) {
                                load_cluster_specific_data_and_nsg_mmap(cluster_id, ctx_mmap_.query_dim, 
                                    ctx_mmap_.cluster_data_map, ctx_mmap_.id_mapping_map, 
                                    ctx_mmap_.cluster_nsg_indices, ctx_mmap_.prefix);
                            } 
                        }
                    }

                    std::priority_queue<std::pair<float, unsigned>> local_queue;


                    search_nsg_cluster_mmap(cluster_id, 
                                            query_data + i * ctx_mmap_.query_dim,
                                            local_queue, current_min_max_dist, 
                                            query_early_stopped,
                                            ctx_mmap_);

                    // 合并结果
                    {
                        std::lock_guard<std::mutex> guard(queue_mutex);
                        merge_topk_queue(topk_queue, local_queue, ctx_mmap_.k);
                        current_min_max_dist = topk_queue.top().first;
                    }
                }

                // 处理搜索结果
                results[i] = process_search_results(topk_queue, ctx_mmap_.k);
                
                // 计算recall
                int query_correct_count = calculate_query_recall(results[i], ground_truth_set);
                recalls[i] = static_cast<double>(query_correct_count) / ground_truth_set.size();
                local_correct += query_correct_count;
                local_total += ground_truth_set.size();
            }

            {
                total_correct_ += local_correct;
                total_ground_truth_ += local_total;
            }
        }

        auto end_time_search = std::chrono::high_resolution_clock::now();
        total_search_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time_search - start_time_search).count();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during search: " << e.what() << std::endl;
        return false;
    }
}
*/

/*
// Old version 2 (batch but parallel for just on cluster batch)
bool IndexSearcher::search_mmap(const float* query_data,
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
        omp_set_num_threads(4);

        {
            int local_correct = 0;
            int local_total = 0;

            // #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < query_num; i++) {
                auto query_start = std::chrono::high_resolution_clock::now();
                
                // 在HNSW图上搜索并获取排序后的cluster
                auto sorted_clusters = search_hnsw_and_sort_clusters_mmap(query_data + i * ctx_mmap_.query_dim, nprobe);

                std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
                std::priority_queue<std::pair<float, unsigned>> topk_queue;
                float current_min_max_dist = std::numeric_limits<float>::max();
                bool query_early_stopped = false;

                // 分batch并行处理clusters
                const size_t batch_size = 4;
                for (size_t batch_start = 0; batch_start < sorted_clusters.size(); batch_start += batch_size) {
                    size_t batch_end = std::min(batch_start + batch_size, sorted_clusters.size());

                    // 使用线程本地存储收集batch结果
                    std::vector<std::priority_queue<std::pair<float, unsigned>>> thread_local_queues(batch_end - batch_start);

                    #pragma omp parallel for
                    for (size_t cluster_offset = 0; cluster_offset < (batch_end - batch_start); ++cluster_offset) {
                        size_t cluster_idx = batch_start + cluster_offset;

                        if (query_early_stopped) continue;

                        faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;

                        // cluster load: critical section
                        {
                            #pragma omp critical(cluster_load)
                            {
                                if (!ctx_mmap_.cluster_nsg_indices.count(cluster_id)) {
                                    load_cluster_specific_data_and_nsg_mmap(cluster_id, ctx_mmap_.query_dim, 
                                        ctx_mmap_.cluster_data_map, ctx_mmap_.id_mapping_map, 
                                        ctx_mmap_.cluster_nsg_indices, ctx_mmap_.prefix);
                                }
                            }
                        }

                        // cluster search - 直接存储到线程本地队列
                        search_nsg_cluster_mmap(cluster_id, 
                                                query_data + i * ctx_mmap_.query_dim,
                                                thread_local_queues[cluster_offset], 
                                                current_min_max_dist, 
                                                query_early_stopped,
                                                ctx_mmap_);
                    }

                    // 批量合并所有线程本地队列到全局队列
                    for (auto& local_queue : thread_local_queues) {
                        merge_topk_queue(topk_queue, local_queue, ctx_mmap_.k);
                    }
                    
                    // 更新当前最小最大距离
                    if (!topk_queue.empty()) {
                        current_min_max_dist = topk_queue.top().first;
                    }

                    // if (query_early_stopped) break;
                }

                // 处理搜索结果
                results[i] = process_search_results(topk_queue, ctx_mmap_.k);
                
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
*/

} // namespace CNNS
