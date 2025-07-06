#!/bin/bash

# CONFIGURABLE
CGROUP_NAME="mymem"
MEM_LIMIT_MB=512
MEM_LIMIT=$((MEM_LIMIT_MB * 1024 * 1024))
CGROUP_PATH="/sys/fs/cgroup/memory/$CGROUP_NAME"
MONITOR_INTERVAL=1  # seconds

# 清理旧的 cgroup
cleanup() {
    echo "Cleaning up..."
    if [ -d "$CGROUP_PATH" ]; then
        sudo cgdelete -g memory:/$CGROUP_NAME 2>/dev/null || true
    fi
}
cleanup

# 创建新 cgroup
sudo cgcreate -g memory:/$CGROUP_NAME || { echo "Failed to create cgroup"; exit 1; }

# 设置内存限制，只设置 memory.limit_in_bytes
sudo cgset -r memory.limit_in_bytes=$MEM_LIMIT $CGROUP_NAME

# 启动程序（直接在 cgroup 中运行）
echo "Launching program in cgroup $CGROUP_NAME with $MEM_LIMIT_MB MB memory..."
LOGFILE="program_output.log"
sudo cgexec -g memory:$CGROUP_NAME "$@" > >(tee "$LOGFILE") 2>&1 &
MAIN_PID=$!

# 获取进程组 ID（用于多线程监控）
PGID=$(ps -o pgid= $MAIN_PID | tr -d ' ')
echo "Main PID: $MAIN_PID, PGID: $PGID"

# 监控内存使用
monitor_memory() {
    echo "Monitoring memory usage for PGID=$PGID (cgroup=$CGROUP_NAME)..."
    while ps -p $MAIN_PID > /dev/null 2>&1; do
        USAGE=$(cat "$CGROUP_PATH/memory.usage_in_bytes" 2>/dev/null || echo 0)
        MAX_USAGE=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes" 2>/dev/null || echo 0)
        echo "[$(date +%T)] Current: $((USAGE / 1024 / 1024))MB | Peak: $((MAX_USAGE / 1024 / 1024))MB / ${MEM_LIMIT_MB}MB"
        
        if grep -q "oom_kill 1" "$CGROUP_PATH/memory.oom_control" 2>/dev/null; then
            echo "⚠️ OOM killer was triggered!"
        fi
        sleep $MONITOR_INTERVAL
    done
}
monitor_memory &

# 等待程序结束
wait $MAIN_PID
MONITOR_PID=$!
wait $MONITOR_PID 2>/dev/null

# OOM 检查
if dmesg | grep -q -i 'oom'; then
    echo "⚠️ Detected OOM messages in dmesg:"
    dmesg | grep -i 'oom' | tail -n 10
fi

# 查看最终内存
FINAL_PEAK=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes")
echo "Final peak memory usage: $((FINAL_PEAK / 1024 / 1024))MB"

# 清理
cleanup
echo "✅ Program complete."