#!/bin/bash

# 定义输出日志文件
LOG_FILE="search_results.log"

# 定义参数数组
nprobe_list=(50 100 200 500 1000)
L_list=(1 5 10 20 50 100 200)
K=1  # 固定参数
threads=64  # 固定参数

# 固定路径参数
query_file="/data1/clz/dataset/bigann/bigann_query.bvecs"
file_type="bvecs"
ground_truth="/data1/clz/dataset/bigann/gnd/idx_100M_k1.ivecs"
index_prefix="bigann100M_100cluster_1000centroid"

# 清空或创建日志文件
echo "===== Search Benchmark Results =====" > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"
echo "===================================" >> "$LOG_FILE"

# 循环运行所有参数组合
for L in "${L_list[@]}"; do
    for nprobe in "${nprobe_list[@]}"; do
        # 运行命令
        echo "Running with nprobe=$nprobe, K=$K, L=$L, threads=$threads"
        output=$(./test_search_mmap "$query_file" "$file_type" "$ground_truth" "$nprobe" "$K" "$L" "$index_prefix" "$threads" 2>&1)
        
        # 提取最后5行中的前3行
        stats=$(echo "$output" | tail -n 5 | head -n 3)
        
        # 将参数和结果写入日志文件
        echo "Parameters: nprobe=$nprobe, K=$K, L=$L" >> "$LOG_FILE"
        echo "$stats" >> "$LOG_FILE"
        echo "-----------------------------------" >> "$LOG_FILE"
        
        # 打印到控制台以便实时查看
        echo "$stats"
        echo "-----------------------------------"
    done
done

echo "All tests completed. Results saved to $LOG_FILE"