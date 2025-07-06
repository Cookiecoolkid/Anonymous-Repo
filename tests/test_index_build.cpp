#include "index_build.h"
#include "statistics.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 14) {
        std::cerr << "Usage: " << argv[0] 
                  << " <data_file> <data_format> <n_clusters> <m_centroids> <K> <L> <iter> <S> <R> <L_nsg> <R_nsg> <C_nsg> <prefix>" << std::endl;
        std::cerr << "  data_format: data file format (fvecs or bvecs)" << std::endl;
        std::cerr << "  K: number of neighbors in NNDescent graph (default: 100)" << std::endl;
        std::cerr << "  L: number of candidates in NNDescent graph (default: 100)" << std::endl;
        std::cerr << "  iter: number of iterations (default: 10)" << std::endl;
        std::cerr << "  S: size of candidate set (default: 10)" << std::endl;
        std::cerr << "  R: maximum degree of each node (default: 100)" << std::endl;
        std::cerr << "  L_nsg: number of candidates in NSG (default: 20)" << std::endl;
        std::cerr << "  R_nsg: maximum degree of each node in NSG (default: 20)" << std::endl;
        std::cerr << "  C_nsg: number of candidates in NSG (default: 500)" << std::endl;
        std::cerr << "  prefix: prefix directory for output files" << std::endl;
        return 1;
    }

    // 解析参数
    std::string data_file = argv[1];
    std::string data_format_str = argv[2];
    int n_clusters = std::atoi(argv[3]);
    int m_centroids = std::atoi(argv[4]);
    int k_nndescent = std::atoi(argv[5]);
    int l_nndescent = std::atoi(argv[6]);
    int iter = std::atoi(argv[7]);
    int s = std::atoi(argv[8]);
    int r = std::atoi(argv[9]);
    int L_nsg = std::atoi(argv[10]);
    int R_nsg = std::atoi(argv[11]);
    int C_nsg = std::atoi(argv[12]);
    std::string prefix = argv[13];

    // 解析数据格式
    CNNS::DataFormat data_format;
    if (data_format_str == "fvecs") {
        data_format = CNNS::DataFormat::FVECS;
    } else if (data_format_str == "bvecs") {
        data_format = CNNS::DataFormat::BVECS;
    } else {
        std::cerr << "Error: data_format must be 'fvecs' or 'bvecs'" << std::endl;
        return 1;
    }

    // 设置默认值
    if (k_nndescent == -1) k_nndescent = 100;
    if (l_nndescent == -1) l_nndescent = 100;
    if (iter == -1) iter = 10;
    if (s == -1) s = 10;
    if (r == -1) r = 100;
    if (L_nsg == -1) L_nsg = 20;
    if (R_nsg == -1) R_nsg = 20;
    if (C_nsg == -1) C_nsg = 500;

    std::cout << "Parameters:" << std::endl
              << "  NNDescent:" << std::endl
              << "    K: " << k_nndescent << std::endl
              << "    L: " << l_nndescent << std::endl
              << "    iter: " << iter << std::endl
              << "    S: " << s << std::endl
              << "    R: " << r << std::endl
              << "  NSG:" << std::endl
              << "    L: " << L_nsg << std::endl
              << "    R: " << R_nsg << std::endl
              << "    C: " << C_nsg << std::endl;

    try {
        // 创建统计对象
        CNNS::Statistics stats;
        
        // 创建 IndexBuilder 实例
        CNNS::IndexBuilder<float> builder(prefix, n_clusters, m_centroids,
                                        k_nndescent, l_nndescent, iter, s, r,
                                        L_nsg, R_nsg, C_nsg);

        // 构建索引
        /*
        if (!builder.build(data_file, true, &stats)) {
            std::cerr << "Failed to build index" << std::endl;
            return 1;
        }
        */

        if (!builder.build_mmap(data_file, data_format, &stats)) {
            std::cerr << "Failed to build index" << std::endl;
            return 1;
        }

        // 输出详细的统计信息
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "BUILD STATISTICS SUMMARY" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        stats.print_build_stats();
        
        // 保存统计信息到文件，文件名包含关键参数信息
        std::string stats_file = prefix + "/build_statistics_clusters" + 
                                std::to_string(n_clusters) + "_K" + std::to_string(k_nndescent) + 
                                "_Lnsg" + std::to_string(L_nsg) + "_Rnsg" + std::to_string(R_nsg) + 
                                "_Cnsg" + std::to_string(C_nsg) + ".txt";
        stats.save_to_file(stats_file);
        std::cout << "\nStatistics saved to: " << stats_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
