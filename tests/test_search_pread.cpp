#include <index_search.h>
#include <aux_util.h>
#include "statistics.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <path_to_ground_truth> <nprobe> <search_K> <search_L> <prefix>" << std::endl;
        std::cerr << "  nprobe: number of clusters to search (default: 50)" << std::endl;
        std::cerr << "  search_K: number of neighbors to search in NSG (default: 100)" << std::endl;
        std::cerr << "  search_L: number of candidates in NSG search (default: 100)" << std::endl;
        std::cerr << "  prefix: directory prefix for all data files" << std::endl;
        return 1;
    }

    try {
        // 创建统计对象
        CNNS::Statistics stats;
        
        // 创建搜索器实例，使用pread版本
        CNNS::IndexSearcher searcher(argv[6], true, atoi(argv[4]), atoi(argv[5]));

        // 初始化搜索器，使用mmap版本
        if (!searcher.initialize_mmap(argv[1], argv[2])) {
            std::cerr << "Error initializing mmap searcher" << std::endl;
            return 1;
        }

        // 加载查询数据
        unsigned query_num;
        unsigned query_dim;
        std::vector<float> query_data = CNNS::load_fvecs(argv[1], query_num, query_dim);
        std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(argv[2]);
        int nprobe = atoi(argv[3]);

        std::cout << "query_num: " << query_num 
                  << " query_dim: " << query_dim
                  << " nprobe: " << nprobe 
                  << " search_K: " << searcher.get_search_K()
                  << " search_L: " << searcher.get_search_L()
                  << " k: " << searcher.get_k() << std::endl;

        // 执行搜索
        std::vector<std::vector<unsigned>> results;
        std::vector<double> recalls;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!searcher.search_pread(query_data.data(), query_num, ground_truth, nprobe, results, recalls, &stats)) {
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

        // 输出每个查询的recall
        // std::cout << "\nPer-query recall:" << std::endl;
        // for (size_t i = 0; i < recalls.size() - 66; ++i) {
        //     std::cout << "Query " << i << ": " << recalls[i] << std::endl;
        // }
        
        // 输出基本结果
        std::cout << "Total Search Time: " << total_time << " seconds" << std::endl;
        std::cout << "Recall Rate: " << recall_rate << std::endl;

        // 输出详细的统计信息
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "SEARCH STATISTICS SUMMARY" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        stats.print_search_stats();
        stats.print_system_stats();
        
        // 保存统计信息到文件
        std::string stats_file = std::string(argv[6]) + "/search_pread_statistics.txt";
        stats.save_to_file(stats_file);
        std::cout << "\nStatistics saved to: " << stats_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
