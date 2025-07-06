#include "index_search.h"
#include "aux_util.h"
#include "data_load.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace CNNS {

void IndexSearcher::search_nsg_cluster_pread(
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& local_queue,
    float& current_min_max_dist,
    bool& query_early_stopped,
    SearchContextMMap& ctx_mmap) {

    efanna2e::IndexNSG* nsg_index = ctx_mmap.cluster_nsg_indices.at(cluster_id);
    const MappingMMap& mapping_info = ctx_mmap.id_mapping_map.at(cluster_id);
    
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", ctx_mmap.search_L);
    paras.Set<unsigned>("P_search", ctx_mmap.search_L);
    paras.Set<unsigned>("K_search", ctx_mmap.search_K);
    std::vector<unsigned> tmp(ctx_mmap.search_K);
    std::vector<float> distances(ctx_mmap.search_K);

    // 使用Search_mmap_with_dist_pread进行搜索，它会同时返回距离
    float cluster_min_distance = std::numeric_limits<float>::max(); // 暂时不使用，但为后续优化做准备
    nsg_index->Search_mmap_with_dist_pread(query_data, nullptr, ctx_mmap.search_K, cluster_id, ctx_mmap.prefix, paras, tmp.data(), distances.data(), cluster_min_distance);

    // 更新本地优先队列
    for (int m_loop = 0; m_loop < ctx_mmap.search_K; m_loop++) {
        unsigned local_id = tmp[m_loop];
        if (local_id >= mapping_info.length) continue;
        unsigned global_id = mapping_info.data[local_id];
        float dist = distances[m_loop];
        
        if (local_queue.size() < ctx_mmap.k) {
            local_queue.push({dist, global_id});
        } else if (dist < local_queue.top().first) {
            local_queue.pop();
            local_queue.push({dist, global_id});
        }
    }

    // 如果当前cluster的最小距离大于等于当前topk中的最大距离，且队列已满，则提前停止
    if (local_queue.size() >= ctx_mmap.k && local_queue.top().first >= current_min_max_dist) {
        query_early_stopped = true;
    }
}

bool IndexSearcher::search_pread(const float* query_data,
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

        {
            int local_correct = 0;
            int local_total = 0;
            std::mutex queue_mutex;

            for (size_t i = 0; i < query_num; i++) {
                auto query_start = std::chrono::high_resolution_clock::now();
                
                // 在HNSW图上搜索并获取排序后的cluster
                std::map<faiss::idx_t, faiss::idx_t> cluster_nearest_point_ids;
                auto sorted_clusters = search_hnsw_and_sort_clusters_mmap(query_data + i * ctx_mmap_.query_dim, nprobe, cluster_nearest_point_ids);

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
                                load_cluster_nsg_mmap(cluster_id, ctx_mmap_.query_dim, 
                                    ctx_mmap_.id_mapping_map, ctx_mmap_.cluster_nsg_indices, 
                                    ctx_mmap_.prefix);
                            } 
                        }
                    }

                    std::priority_queue<std::pair<float, unsigned>> local_queue;

                    search_nsg_cluster_pread(cluster_id, 
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

                std::cout << "query " << i << " recall: " << recalls[i] << std::endl;

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
}

