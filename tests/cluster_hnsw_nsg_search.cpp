#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <index_nsg.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <algorithm>
#include <dirent.h>
#include <regex>
#include <unordered_map>
#include <aux_util.h>


int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <path_to_ground_truth> <nprobe> <search_K> <search_L> <prefix>" << std::endl;
        std::cerr << "  nprobe: number of clusters to search (default: 100)" << std::endl;
        std::cerr << "  search_K: number of neighbors to search in NSG (default: 100)" << std::endl;
        std::cerr << "  search_L: number of candidates in NSG search (default: 100)" << std::endl;
        std::cerr << "  prefix: directory prefix for all data files" << std::endl;
        return 1;
    }

    // 1. 加载查询数据和ground truth
    unsigned query_num, query_dim;
    std::vector<float> query_data = CNNS::load_fvecs(argv[1], query_num, query_dim);
    std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(argv[2]);
    int nprobe = atoi(argv[3]); // 表示在质心图上搜索的邻居数（越大，最后搜索的cluster越多，召回率越高）
    int search_K = atoi(argv[4]); // NSG搜索的邻居数
    int search_L = atoi(argv[5]); // NSG搜索的候选数
    std::string prefix = argv[6];

    int k = 100; // 最终返回的近邻数

    std::cout << "query_num: " << query_num << " query_dim: " << query_dim 
              << " nprobe: " << nprobe 
              << " search_K: " << search_K 
              << " search_L: " << search_L
              << " k: " << k << std::endl;

    // 加载质心
    int n_clusters, m;
    unsigned centroids_dim;
    std::vector<float> centroids = CNNS::load_centroids(prefix + "/centroids.fvecs", n_clusters, m, centroids_dim);
    if (centroids_dim != query_dim) {
        std::cerr << "Dimension mismatch between data and centroids" << std::endl;
        return 1;
    }

    // 创建HNSW图索引
    faiss::IndexHNSWFlat* index_hnsw = new faiss::IndexHNSWFlat(query_dim, 32, faiss::METRIC_L2);
    index_hnsw->add(n_clusters * (m + 1), centroids.data());

    // 加载所有NSG图
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, float*> cluster_data_map;
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;
    DIR* dir;
    struct dirent* ent;
    std::regex pattern("nsg_(\\d+)\\.nsg");


    if ((dir = opendir((prefix + "/nsg_graph").c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            std::smatch matches;
            if (std::regex_match(filename, matches, pattern)) {
                int cluster_id = std::stoi(matches[1]);
                
                // 加载cluster数据
                std::string cluster_filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
                unsigned points_num, dim;
                std::ifstream in(cluster_filename, std::ios::binary);
                if (!in.is_open()) {
                    std::cerr << "Error: Cannot open cluster file " << cluster_filename << std::endl;
                    continue;
                }
                in.read((char*)&dim, 4);
                in.seekg(0, std::ios::end);
                size_t fsize = in.tellg();
                points_num = fsize / ((dim + 1) * 4);

                float* cluster_data = new float[points_num * dim * sizeof(float)];
                in.seekg(0, std::ios::beg);
                for (size_t i = 0; i < points_num; i++) {
                    in.seekg(4, std::ios::cur);
                    in.read((char*)(cluster_data + i * dim), dim * 4);
                }
                in.close();

                // 加载ID映射
                std::string mapping_filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
                std::ifstream mapping_file(mapping_filename, std::ios::binary);
                if (!mapping_file.is_open()) {
                    std::cerr << "Error: Cannot open mapping file " << mapping_filename << std::endl;
                    delete[] cluster_data;
                    continue;
                }
                std::vector<faiss::idx_t> id_mapping(points_num);
                mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
                mapping_file.close();

                // 创建并加载NSG索引
                efanna2e::IndexNSG* nsg_index = new efanna2e::IndexNSG(dim, points_num, efanna2e::L2, nullptr);
                std::string nsg_filename = prefix + "/nsg_graph/" + filename;
                try {
                    nsg_index->Load(nsg_filename.c_str());
                    // std::cout << "Successfully loaded NSG from " << nsg_filename << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error loading NSG from " << nsg_filename << ": " << e.what() << std::endl;
                    delete nsg_index;
                    delete[] cluster_data;
                    continue;
                }

                cluster_nsg_indices[cluster_id] = nsg_index;
                cluster_data_map[cluster_id] = cluster_data;
                id_mapping_map[cluster_id] = id_mapping;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Cannot open nsg_graph directory" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << cluster_nsg_indices.size() << " NSG indices" << std::endl;

    // 搜索过程
    int correct = 0;
    int total = 0;
    auto start_time_search = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < query_num; i++) {
        // 在HNSW图上搜索得到nprobe个点
        std::vector<float> query_distances(nprobe);
        std::vector<faiss::idx_t> query_labels(nprobe);
        index_hnsw->search(1, query_data.data() + i * query_dim, nprobe, 
                          query_distances.data(), query_labels.data());

        // 统计每个cluster包含的样本点数量
        std::map<faiss::idx_t, int> cluster_sample_count;
        for (int j = 0; j < nprobe; ++j) {
            faiss::idx_t point_id = query_labels[j];
            faiss::idx_t cluster_id = point_id / (m + 1);
            cluster_sample_count[cluster_id]++;
        }

        // // 输出选中的cluster ID
        // std::cout << "Query " << i << " selected cluster ids: ";
        // for (auto cluster_id : selected_clusters) {
        //     std::cout << cluster_id << " ";
        // }
        // std::cout << std::endl;

        // // 输出ground truth
        // std::cout << "Query " << i << " ground truth: ";
        // for (auto gt : ground_truth[i]) {
        //     std::cout << gt << " ";
        // }
        // std::cout << std::endl;
        // 将cluster按样本点数量排序
        std::vector<std::pair<faiss::idx_t, int>> sorted_clusters;
        for (const auto& pair : cluster_sample_count) {
            sorted_clusters.push_back(pair);
        }
        std::sort(sorted_clusters.begin(), sorted_clusters.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        // 检查ground truth分布在哪些cluster
        std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
        std::cout << "Ground truth distribution in clusters:" << std::endl;
        for (auto& [cluster_id, mapping] : id_mapping_map) {
            for (size_t local_id = 0; local_id < mapping.size(); local_id++) {
                unsigned global_id = mapping[local_id];
                if (ground_truth_set.count(global_id)) {
                    std::cout << "GT " << global_id << " in cluster " << cluster_id 
                              << " (local_id: " << local_id << ")" << std::endl;
                }
            }
        }

        // 使用map存储全局ID到距离的映射
        std::unordered_map<unsigned, float> global_id_to_dist;
        float current_min_max_dist = std::numeric_limits<float>::max();

        // 在NSG上搜索得到对应的点
        for (const auto& [cluster_id, sample_count] : sorted_clusters) {
            if (cluster_nsg_indices.find(cluster_id) == cluster_nsg_indices.end()) {
                std::cout << "Warning: cluster " << cluster_id << " not found in NSG indices" << std::endl;
                continue;
            }

            efanna2e::IndexNSG* nsg_index = cluster_nsg_indices[cluster_id];
            float* cluster_data = cluster_data_map[cluster_id];
            efanna2e::Parameters paras;
            paras.Set<unsigned>("L_search", search_L);
            paras.Set<unsigned>("P_search", search_L);
            paras.Set<unsigned>("K_search", search_K);

            std::vector<unsigned> tmp(search_K);
            nsg_index->Search(query_data.data() + i * query_dim, 
                            cluster_data, search_K, paras, tmp.data());

            // 计算当前cluster中最小距离
            float cluster_min_dist = std::numeric_limits<float>::max();
            for (int m = 0; m < search_K; m++) {
                unsigned local_id = tmp[m];
                if (local_id >= id_mapping_map[cluster_id].size()) {
                    std::cerr << "Error: local_id " << local_id << " out of range for cluster " << cluster_id 
                              << " (size: " << id_mapping_map[cluster_id].size() << ")" << std::endl;
                    continue;
                }
                unsigned global_id = id_mapping_map[cluster_id][local_id];
                
                // 计算实际距离
                float dist = 0;
                for (unsigned d = 0; d < query_dim; d++) {
                    float diff = query_data[i * query_dim + d] - cluster_data[local_id * query_dim + d];
                    dist += diff * diff;
                }
                cluster_min_dist = std::min(cluster_min_dist, dist);

                // 更新或添加距离
                if (global_id_to_dist.count(global_id)) {
                    global_id_to_dist[global_id] = std::min(global_id_to_dist[global_id], dist);
                } else {
                    global_id_to_dist[global_id] = dist;
                }
            }

            // 如果当前cluster的最小距离大于等于之前所有cluster中最大距离的最小值，则停止搜索
            if (cluster_min_dist >= current_min_max_dist) {
                break;
            }

            // 更新当前最小最大距离
            if (!global_id_to_dist.empty()) {
                std::vector<float> distances;
                distances.reserve(global_id_to_dist.size());
                for (const auto& pair : global_id_to_dist) {
                    distances.push_back(pair.second);
                }
                std::sort(distances.begin(), distances.end());
                current_min_max_dist = std::min(current_min_max_dist, distances[std::min(k - 1, (int)distances.size() - 1)]);
            }
        }

        // 将map转换为vector并按距离排序
        std::vector<std::pair<float, unsigned>> all_results;
        all_results.reserve(global_id_to_dist.size());
        for (auto& [global_id, dist] : global_id_to_dist) {
            all_results.push_back({dist, global_id});
        }

        // 使用partial_sort进行部分排序
        std::partial_sort(all_results.begin(), all_results.begin() + std::min(k, (int)all_results.size()), 
                         all_results.end());

        // 选择前k个结果
        std::vector<unsigned> final_results;
        final_results.reserve(k);
        for (int m = 0; m < k && m < (int)all_results.size(); m++) {
            final_results.push_back(all_results[m].second);
        }

        // 输出预测结果
        std::cout << "Query " << i << " predicted neighbors: ";
        for (auto pred : final_results) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;

        // 计算recall rate
        int query_correct = 0;
        for (unsigned id : final_results) {
            if (ground_truth_set.count(id)) {
                correct++;
                query_correct++;
            }
        }
        total += ground_truth_set.size();
        
        // 输出每个查询的recall
        std::cout << "Query " << i << " recall: " << static_cast<double>(query_correct) / ground_truth_set.size() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    auto end_time_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_time = end_time_search - start_time_search;
    std::cout << "Search Time: " << search_time.count() << " seconds" << std::endl;
    std::cout << "correct: " << correct << " total: " << total << std::endl;
    std::cout << "Recall Rate: " << static_cast<double>(correct) / total << std::endl;

    // 清理资源
    delete index_hnsw;
    for (auto& pair : cluster_nsg_indices) {
        delete pair.second;
    }
    for (auto& pair : cluster_data_map) {
        delete[] pair.second;
    }

    return 0;
}
