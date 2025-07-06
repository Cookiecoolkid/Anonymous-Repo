#include <index_search.h>
#include <aux_util.h>
#include "statistics.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <query_format> <path_to_ground_truth> <nprobe> <search_K> <search_L> <prefix> <num_threads>" << std::endl;
        std::cerr << "  query_format: query data file format (fvecs or bvecs)" << std::endl;
        std::cerr << "  nprobe: number of clusters to search (default: 50)" << std::endl;
        std::cerr << "  search_K: number of neighbors to search in NSG (default: 100)" << std::endl;
        std::cerr << "  search_L: number of candidates in NSG search (default: 100 / should >= search_K)" << std::endl;
        std::cerr << "  prefix: directory prefix for all data files" << std::endl;
        std::cerr << "  num_threads: number of threads to use (-1 for auto-detect)" << std::endl;
        return 1;
    }

    try {
        // 解析数据格式
        std::string query_format_str = argv[2];
        CNNS::DataFormat query_format;
        if (query_format_str == "fvecs") {
            query_format = CNNS::DataFormat::FVECS;
        } else if (query_format_str == "bvecs") {
            query_format = CNNS::DataFormat::BVECS;
        } else {
            std::cerr << "Error: query_format must be 'fvecs' or 'bvecs'" << std::endl;
            return 1;
        }
        
        // 创建统计对象
        CNNS::Statistics stats;
        int search_K = atoi(argv[5]);
        int search_L = atoi(argv[6]);
        int num_threads = atoi(argv[8]);
        
        // 创建搜索器实例，使用mmap版本
        CNNS::IndexSearcher searcher(argv[7], true, search_K, search_L, search_K);

        // 初始化搜索器，使用mmap版本
        if (!searcher.initialize_mmap(argv[1], argv[3])) {
            std::cerr << "Error initializing mmap searcher" << std::endl;
            return 1;
        }

        // 加载查询数据
        unsigned query_num;
        unsigned query_dim;
        std::vector<float> query_data;
        
        // 根据格式加载查询数据
        if (query_format == CNNS::DataFormat::FVECS) {
            query_data = CNNS::load_fvecs(argv[1], query_num, query_dim);
        } else if (query_format == CNNS::DataFormat::BVECS) {
            auto bvec_data = CNNS::load_bvecs(argv[1], query_num, query_dim);
            // 将uint8转换为float
            query_data.resize(static_cast<size_t>(query_num) * static_cast<size_t>(query_dim));
            for (size_t i = 0; i < static_cast<size_t>(query_num) * static_cast<size_t>(query_dim); ++i) {
                query_data[i] = static_cast<float>(bvec_data[i]);
            }
        }
        
        std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(argv[3]);
        int nprobe = atoi(argv[4]);

        std::cout << "query_num: " << query_num 
                  << " query_dim: " << query_dim
                  << " nprobe: " << nprobe 
                  << " search_K: " << searcher.get_search_K()
                  << " search_L: " << searcher.get_search_L()
                  << " k: " << searcher.get_k()
                  << " num_threads: " << num_threads << std::endl;

        // 执行搜索
        std::vector<std::vector<unsigned>> results;
        std::vector<double> recalls;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!searcher.search_mmap(query_data.data(), query_num, ground_truth, nprobe, results, recalls, &stats, num_threads)) {
            std::cerr << "Error during search" << std::endl;
            return 1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> search_time = end_time - start_time;

        // 获取详细的内存统计信息并记录到Statistics对象中
        size_t peak_from_proc = CNNS::getPeakMemoryFromProc();
        size_t peak_from_rusage = CNNS::getProcessPeakRSS();
        size_t peak_physical_memory = CNNS::getPeakPhysicalMemoryKB();
        CNNS::PageFaultStats page_fault_stats = CNNS::getPageFaultStats();
        
        // 记录详细的内存统计
        stats.record_detailed_memory_stats(peak_from_proc, peak_from_rusage, 
                                          peak_physical_memory, page_fault_stats.minor, 
                                          page_fault_stats.major);

        // 获取统计信息
        double total_time, recall_rate;
        searcher.get_stats(total_time, recall_rate);

        
        // 输出基本结果
        std::cout << "Total Search Time: " << total_time << " seconds" << std::endl;
        std::cout << "Recall Rate: " << recall_rate << std::endl;

        // 使用新的表格输出方式
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "SEARCH STATISTICS TABLE" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // 使用search_L作为L参数，search_K
        stats.print_table_stats(search_L, search_K);
        
        // 保存统计信息到文件，文件名包含参数信息
        std::string stats_file = std::string(argv[7]) + "/search_statistics_nprobe" + 
                                std::to_string(nprobe) + "_K" + std::to_string(search_K) + 
                                "_L" + std::to_string(search_L) + "_threads" + std::to_string(num_threads) + ".txt";
        stats.save_to_file(stats_file);
        std::cout << "\nStatistics saved to: " << stats_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
