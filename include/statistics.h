#pragma once

#include <index_nsg.h>
#include <util.h>
#include "data_load.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <chrono>

namespace CNNS {

class Statistics {
public:
    Statistics() = default;
    ~Statistics() = default;

    // 构建阶段统计信息
    struct BuildStats {
        // 时间统计（毫秒）
        double ivf_build_time = 0.0;
        double hnsw_build_time = 0.0;
        double nndescent_build_time = 0.0;
        double nsg_build_time = 0.0;
        double total_build_time = 0.0;
        
        // 时间占比（百分比）
        double ivf_percent = 0.0;
        double hnsw_percent = 0.0;
        double nndescent_percent = 0.0;
        double nsg_percent = 0.0;
    };

    // 搜索阶段统计信息
    struct SearchStats {
        // 时间统计
        double total_search_time = 0.0;
        double latency_mean = 0.0;
        // 99.9% Query Latency
        double latency_99_9 = 0.0;
        
        // QPS统计
        double qps = 0.0;
        
        // 性能统计
        double total_recall = 0.0;
        double min_recall = 0.0;
        double max_recall = 0.0;

        // IO - 只保留n_ios，移除avg_io_time
        size_t n_ios = 0;
        
        // 查询数量
        unsigned query_count = 0;
        
        // 图搜索跳数统计
        unsigned graph_abstraction_nhops = 0;  // HNSW图抽象搜索跳数
        unsigned local_graph_nhops = 0;        // 本地图搜索跳数
    };

    // 系统统计信息
    struct SystemStats {
        // 新增内存统计
        size_t peak_memory_from_proc = 0;      // 从/proc/self/status获取的峰值内存
        size_t peak_memory_from_rusage = 0;    // 从getrusage获取的峰值RSS
        size_t peak_physical_memory = 0;       // 峰值物理内存
        size_t page_faults_minor = 0;          // 次要页面错误
        size_t page_faults_major = 0;          // 主要页面错误
        size_t total_page_faults = 0;          // 总页面错误
        
        // 线程信息
        unsigned num_threads = 0;
    };

    // 成员变量
    BuildStats build_stats;
    SearchStats search_stats;
    SystemStats system_stats;

    // 构建统计方法
    void record_ivf_build_time(double time_ms);
    void record_hnsw_build_time(double time_ms);
    void record_nndescent_build_time(double time_ms);
    void record_nsg_build_time(double time_ms);
    void record_total_build_time(double time_ms);
    void calculate_build_percentages();
    
    // 搜索统计方法
    void record_search_time(double time_ms);
    void record_query_results(const std::vector<double>& recalls);
    void record_cache_stats(unsigned hits, unsigned misses);
    void record_qps(unsigned query_count, double total_time);
    void record_latency_percentiles(const std::vector<double>& query_times);
    void record_graph_abstraction_nhops(unsigned nhops);
    void record_local_graph_nhops(unsigned nhops);
    
    // 系统统计方法
    void record_detailed_memory_stats(size_t peak_from_proc, size_t peak_from_rusage, 
                                     size_t peak_physical, size_t page_faults_minor, 
                                     size_t page_faults_major);
    void record_thread_info(unsigned num_threads);
    void record_file_sizes(const std::string& prefix);
    
    // 输出方法
    void print_build_stats() const;
    void print_search_stats() const;
    void print_system_stats() const;
    void print_all_stats() const;
    
    // 新增表格输出方法
    void print_table_header(int K) const;
    void print_table_row(int L, int K) const;
    void print_table_stats(int L, int K) const;
    
    // 保存到文件
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    
    // 重置统计信息
    void reset_build_stats();
    void reset_search_stats();
    void reset_system_stats();
    void reset_all_stats();
};

} // namespace CNNS