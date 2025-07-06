#include "statistics.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace CNNS {

// 构建统计方法
void Statistics::record_ivf_build_time(double time_ms) {
    build_stats.ivf_build_time = time_ms;
}

void Statistics::record_hnsw_build_time(double time_ms) {
    build_stats.hnsw_build_time = time_ms;
}

void Statistics::record_nndescent_build_time(double time_ms) {
    build_stats.nndescent_build_time = time_ms;
}

void Statistics::record_nsg_build_time(double time_ms) {
    build_stats.nsg_build_time = time_ms;
}

void Statistics::record_total_build_time(double time_ms) {
    build_stats.total_build_time = time_ms;
}

void Statistics::calculate_build_percentages() {
    if (build_stats.total_build_time > 0) {
        build_stats.ivf_percent = (build_stats.ivf_build_time / build_stats.total_build_time) * 100;
        build_stats.hnsw_percent = (build_stats.hnsw_build_time / build_stats.total_build_time) * 100;
        build_stats.nndescent_percent = (build_stats.nndescent_build_time / build_stats.total_build_time) * 100;
        build_stats.nsg_percent = (build_stats.nsg_build_time / build_stats.total_build_time) * 100;
    }
}

// 搜索统计方法
void Statistics::record_search_time(double time_ms) {
    search_stats.total_search_time = time_ms;
}

void Statistics::record_query_results(const std::vector<double>& recalls) {
    if (recalls.empty()) return;
    
    search_stats.total_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    search_stats.min_recall = *std::min_element(recalls.begin(), recalls.end());
    search_stats.max_recall = *std::max_element(recalls.begin(), recalls.end());

    for (size_t i = 0; i < recalls.size(); ++i) {
        if (recalls[i] < 0.4) {
            std::cout << "recall: " << recalls[i] << " at index: " << i << std::endl;
        }
    }
}

void Statistics::record_qps(unsigned query_count, double total_time) {
    search_stats.query_count = query_count;
    search_stats.total_search_time = total_time;
    
    if (total_time > 0) {
        search_stats.qps = static_cast<double>(query_count) / total_time;
    }
}

void Statistics::record_latency_percentiles(const std::vector<double>& query_times) {
    if (query_times.empty()) return;
    
    // 计算平均延迟：所有query_times加起来除以query_count
    double total_query_time = 0.0;
    for (const auto& time : query_times) {
        total_query_time += time;
    }
    search_stats.latency_mean = total_query_time / static_cast<double>(query_times.size());
    
    std::vector<double> sorted_times = query_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    // 计算99.9%延迟
    size_t index_99_9 = static_cast<size_t>(sorted_times.size() * 0.999);
    if (index_99_9 >= sorted_times.size()) {
        index_99_9 = sorted_times.size() - 1;
    }
    search_stats.latency_99_9 = sorted_times[index_99_9];
}

void Statistics::record_graph_abstraction_nhops(unsigned nhops) {
    search_stats.graph_abstraction_nhops = nhops;
}

void Statistics::record_local_graph_nhops(unsigned nhops) {
    search_stats.local_graph_nhops = nhops;
}

void Statistics::record_cache_stats(unsigned hits, unsigned misses) {
    // 这个方法现在可以用于记录IO统计信息
    search_stats.n_ios = hits + misses;
}

// 系统统计方法
void Statistics::record_detailed_memory_stats(size_t peak_from_proc, size_t peak_from_rusage, 
                                             size_t peak_physical, size_t page_faults_minor, 
                                             size_t page_faults_major) {
    system_stats.peak_memory_from_proc = peak_from_proc;
    system_stats.peak_memory_from_rusage = peak_from_rusage;
    system_stats.peak_physical_memory = peak_physical;
    system_stats.page_faults_minor = page_faults_minor;
    system_stats.page_faults_major = page_faults_major;
    system_stats.total_page_faults = page_faults_minor + page_faults_major;
}

void Statistics::record_thread_info(unsigned num_threads) {
    system_stats.num_threads = num_threads;
}

void Statistics::record_file_sizes(const std::string& prefix) {
    // 简化版本，只记录基本信息
    // 如果需要文件大小统计，可以使用系统命令或第三方库
    std::cout << "File size calculation for prefix: " << prefix << std::endl;
}

// 输出方法
void Statistics::print_build_stats() const {
    std::cout << "=== Build Statistics ===" << std::endl;
    std::cout << "IVF build time: " << std::fixed << std::setprecision(3) 
              << build_stats.ivf_build_time / 1000.0 << " s (" 
              << std::setprecision(1) << build_stats.ivf_percent << "%)" << std::endl;
    std::cout << "HNSW build time: " << std::fixed << std::setprecision(3) 
              << build_stats.hnsw_build_time / 1000.0 << " s (" 
              << std::setprecision(1) << build_stats.hnsw_percent << "%)" << std::endl;
    std::cout << "NNDescent build time: " << std::fixed << std::setprecision(3) 
              << build_stats.nndescent_build_time / 1000.0 << " s (" 
              << std::setprecision(1) << build_stats.nndescent_percent << "%)" << std::endl;
    std::cout << "NSG build time: " << std::fixed << std::setprecision(3) 
              << build_stats.nsg_build_time / 1000.0 << " s (" 
              << std::setprecision(1) << build_stats.nsg_percent << "%)" << std::endl;
    std::cout << "Total build time: " << std::fixed << std::setprecision(3) 
              << build_stats.total_build_time / 1000.0 << " s" << std::endl;
}

void Statistics::print_search_stats() const {
    std::cout << "=== Search Statistics ===" << std::endl;
    std::cout << "Total search time: " << std::fixed << std::setprecision(3) 
              << search_stats.total_search_time << " s" << std::endl;
    std::cout << "Query count: " << search_stats.query_count << std::endl;
    std::cout << "QPS: " << std::fixed << std::setprecision(2) 
              << search_stats.qps << " queries/s" << std::endl;
    std::cout << "Mean latency: " << std::fixed << std::setprecision(3) 
              << search_stats.latency_mean * 1000.0 << " ms" << std::endl;
    std::cout << "99.9% latency: " << std::fixed << std::setprecision(3) 
              << search_stats.latency_99_9 * 1000.0 << " ms" << std::endl;
    
    std::cout << "\nPerformance Statistics:" << std::endl;
    std::cout << "Average recall: " << std::fixed << std::setprecision(6) 
              << search_stats.total_recall << std::endl;
    std::cout << "Min recall: " << std::fixed << std::setprecision(6) 
              << search_stats.min_recall << std::endl;
    std::cout << "Max recall: " << std::fixed << std::setprecision(6) 
              << search_stats.max_recall << std::endl;
    
    std::cout << "\nIO Statistics:" << std::endl;
    std::cout << "Number of IOs: " << search_stats.n_ios << std::endl;
    std::cout << "Minor page faults: " << system_stats.page_faults_minor << std::endl;
    
    std::cout << "\nGraph Search Statistics:" << std::endl;
    std::cout << "Graph abstraction nhops: " << search_stats.graph_abstraction_nhops << std::endl;
    std::cout << "Local graph nhops: " << search_stats.local_graph_nhops << std::endl;
}

void Statistics::print_system_stats() const {
    std::cout << "=== System Statistics ===" << std::endl;
    
    std::cout << "Memory Statistics:" << std::endl;
    std::cout << "Peak memory from getrusage: " << system_stats.peak_memory_from_rusage << " KB (" 
              << std::fixed << std::setprecision(2) << system_stats.peak_memory_from_rusage / 1024.0 << " MB)" << std::endl;
    std::cout << "Peak physical memory: " << system_stats.peak_physical_memory << " KB (" 
              << std::fixed << std::setprecision(2) << system_stats.peak_physical_memory / 1024.0 << " MB)" << std::endl;
    
    std::cout << "\nPage Fault Statistics:" << std::endl;
    std::cout << "Minor page faults: " << system_stats.page_faults_minor << std::endl;
    std::cout << "Major page faults: " << system_stats.page_faults_major << std::endl;
    std::cout << "Total page faults: " << system_stats.total_page_faults << std::endl;
    
    std::cout << "Number of threads: " << system_stats.num_threads << std::endl;
}

void Statistics::print_all_stats() const {
    print_build_stats();
    std::cout << std::endl;
    print_search_stats();
    std::cout << std::endl;
    print_system_stats();
}

// 保存到文件
void Statistics::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "=== Build Statistics ===" << std::endl;
    file << "ivf_build_time: " << build_stats.ivf_build_time << std::endl;
    file << "hnsw_build_time: " << build_stats.hnsw_build_time << std::endl;
    file << "nndescent_build_time: " << build_stats.nndescent_build_time << std::endl;
    file << "nsg_build_time: " << build_stats.nsg_build_time << std::endl;
    file << "total_build_time: " << build_stats.total_build_time << std::endl;
    file << "ivf_percent: " << build_stats.ivf_percent << std::endl;
    file << "hnsw_percent: " << build_stats.hnsw_percent << std::endl;
    file << "nndescent_percent: " << build_stats.nndescent_percent << std::endl;
    file << "nsg_percent: " << build_stats.nsg_percent << std::endl;
    
    file << "\n=== Search Statistics ===" << std::endl;
    file << "total_search_time: " << search_stats.total_search_time << std::endl;
    file << "query_count: " << search_stats.query_count << std::endl;
    file << "qps: " << search_stats.qps << std::endl;
    file << "latency_mean: " << search_stats.latency_mean * 1000.0 << " ms" << std::endl;
    file << "latency_99_9: " << search_stats.latency_99_9 * 1000.0 << " ms" << std::endl;
    file << "average_recall: " << search_stats.total_recall << std::endl;
    file << "min_recall: " << search_stats.min_recall << std::endl;
    file << "max_recall: " << search_stats.max_recall << std::endl;
    file << "n_ios: " << search_stats.n_ios << std::endl;
    file << "graph_abstraction_nhops: " << search_stats.graph_abstraction_nhops << std::endl;
    file << "local_graph_nhops: " << search_stats.local_graph_nhops << std::endl;
    
    file << "\n=== System Statistics ===" << std::endl;
    file << "peak_memory_from_proc: " << system_stats.peak_memory_from_proc << std::endl;
    file << "peak_memory_from_rusage: " << system_stats.peak_memory_from_rusage << std::endl;
    file << "peak_physical_memory: " << system_stats.peak_physical_memory << std::endl;
    file << "page_faults_minor: " << system_stats.page_faults_minor << std::endl;
    file << "page_faults_major: " << system_stats.page_faults_major << std::endl;
    file << "total_page_faults: " << system_stats.total_page_faults << std::endl;
    file << "num_threads: " << system_stats.num_threads << std::endl;
    
    file.close();
}

void Statistics::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for reading: " << filename << std::endl;
        return;
    }
    
    std::string line;
    std::string section;
    
    while (std::getline(file, line)) {
        if (line.find("===") != std::string::npos) {
            section = line;
            continue;
        }
        
        if (line.empty()) continue;
        
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        // 移除前导空格
        value.erase(0, value.find_first_not_of(" \t"));
        
        // 解析数值
        if (section.find("Build") != std::string::npos) {
            if (key == "ivf_build_time") build_stats.ivf_build_time = std::stod(value);
            else if (key == "hnsw_build_time") build_stats.hnsw_build_time = std::stod(value);
            else if (key == "nndescent_build_time") build_stats.nndescent_build_time = std::stod(value);
            else if (key == "nsg_build_time") build_stats.nsg_build_time = std::stod(value);
            else if (key == "total_build_time") build_stats.total_build_time = std::stod(value);
            else if (key == "ivf_percent") build_stats.ivf_percent = std::stod(value);
            else if (key == "hnsw_percent") build_stats.hnsw_percent = std::stod(value);
            else if (key == "nndescent_percent") build_stats.nndescent_percent = std::stod(value);
            else if (key == "nsg_percent") build_stats.nsg_percent = std::stod(value);
        }
        else if (section.find("Search") != std::string::npos) {
            if (key == "total_search_time") search_stats.total_search_time = std::stod(value);
            else if (key == "query_count") search_stats.query_count = std::stoul(value);
            else if (key == "qps") search_stats.qps = std::stod(value);
            else if (key == "latency_mean") search_stats.latency_mean = std::stod(value);
            else if (key == "latency_99_9") search_stats.latency_99_9 = std::stod(value);
            else if (key == "average_recall") search_stats.total_recall = std::stod(value);
            else if (key == "min_recall") search_stats.min_recall = std::stod(value);
            else if (key == "max_recall") search_stats.max_recall = std::stod(value);
            else if (key == "n_ios") search_stats.n_ios = std::stoull(value);
            else if (key == "graph_abstraction_nhops") search_stats.graph_abstraction_nhops = std::stoul(value);
            else if (key == "local_graph_nhops") search_stats.local_graph_nhops = std::stoul(value);
        }
        else if (section.find("System") != std::string::npos) {
            if (key == "peak_memory_from_proc") system_stats.peak_memory_from_proc = std::stoull(value);
            else if (key == "peak_memory_from_rusage") system_stats.peak_memory_from_rusage = std::stoull(value);
            else if (key == "peak_physical_memory") system_stats.peak_physical_memory = std::stoull(value);
            else if (key == "page_faults_minor") system_stats.page_faults_minor = std::stoull(value);
            else if (key == "page_faults_major") system_stats.page_faults_major = std::stoull(value);
            else if (key == "total_page_faults") system_stats.total_page_faults = std::stoull(value);
            else if (key == "num_threads") system_stats.num_threads = std::stoul(value);
        }
    }
    
    file.close();
}

// 重置统计信息
void Statistics::reset_build_stats() {
    build_stats = BuildStats{};
}

void Statistics::reset_search_stats() {
    search_stats = SearchStats{};
}

void Statistics::reset_system_stats() {
    system_stats = SystemStats{};
}

void Statistics::reset_all_stats() {
    reset_build_stats();
    reset_search_stats();
    reset_system_stats();
}

// 新增表格输出方法
void Statistics::print_table_header(int K) const {
    std::cout << std::setw(4) << "L" 
              << std::setw(12) << "K" 
              << std::setw(12) << "QPS" 
              << std::setw(15) << "Mean Latency" 
              << std::setw(15) << "99.9 Latency" 
              << std::setw(12) << "Total IOs" 
              << std::setw(12) << "Mean Hops" 
              << std::setw(12) << "CPU (s)" 
              << std::setw(12) << "Query Count" 
              << std::setw(15) << "Peak Mem(MB)" 
              << std::setw(12) << "Recall@" << K 
              << std::endl;
}

void Statistics::print_table_row(int L, int K) const {
    // 计算平均跳数
    double mean_hops = 0.0;
    if (search_stats.graph_abstraction_nhops > 0 || search_stats.local_graph_nhops > 0) {
        mean_hops = (search_stats.graph_abstraction_nhops + search_stats.local_graph_nhops) / 2.0;
    }
    
    // 使用total page faults作为IO统计
    size_t total_ios = system_stats.total_page_faults;
    
    std::cout << std::setw(4) << L 
              << std::setw(12) << K 
              << std::setw(12) << std::fixed << std::setprecision(2) << search_stats.qps 
              << std::setw(15) << std::fixed << std::setprecision(2) << (search_stats.latency_mean * 1000.0) 
              << std::setw(15) << std::fixed << std::setprecision(2) << (search_stats.latency_99_9 * 1000.0) 
              << std::setw(12) << total_ios 
              << std::setw(12) << std::fixed << std::setprecision(2) << mean_hops 
              << std::setw(12) << std::fixed << std::setprecision(6) << search_stats.total_search_time 
              << std::setw(12) << search_stats.query_count 
              << std::setw(15) << std::fixed << std::setprecision(2) << (system_stats.peak_memory_from_rusage / 1024.0) 
              << std::setw(12) << std::fixed << std::setprecision(2) << (search_stats.total_recall * 100.0) 
              << std::endl;
}

void Statistics::print_table_stats(int L, int K) const {
    print_table_header(K);
    print_table_row(L, K);
}

} // namespace CNNS
