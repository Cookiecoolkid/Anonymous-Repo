#!/bin/bash

CGROUP_NAME="mymemgroup"
CGROUP_MEM_DIR="/sys/fs/cgroup/memory/$CGROUP_NAME"

# 设置内存限制 (512MB)
MEM_LIMIT=$((512 * 1024 * 1024)) 

# 清理旧的 cgroup
sudo rmdir "$CGROUP_MEM_DIR" 2>/dev/null || true

# 创建新的 cgroup
sudo mkdir -p "$CGROUP_MEM_DIR"

# 设置内存限制 (关键设置)
echo $MEM_LIMIT | sudo tee "$CGROUP_MEM_DIR/memory.limit_in_bytes" > /dev/null

# 启用内存回收机制
echo 1 | sudo tee "$CGROUP_MEM_DIR/memory.use_hierarchy" > /dev/null

# 设置内存压力通知
echo 100 | sudo tee "$CGROUP_MEM_DIR/memory.pressure_level" > /dev/null

# 设置 OOM 控制策略
echo 1 | sudo tee "$CGROUP_MEM_DIR/memory.oom_control" > /dev/null

# 启动程序
"$@" &
PID=$!
echo $PID | sudo tee "$CGROUP_MEM_DIR/tasks" > /dev/null

# 监控内存使用
echo "Monitoring PID $PID with 512MB memory limit..."
while kill -0 $PID 2>/dev/null; do
    echo "[Memory Usage]"
    echo "RSS: $(cat "$CGROUP_MEM_DIR/memory.stat" | grep 'total_rss' | awk '{print $2}') bytes"
    echo "Mapped: $(cat "$CGROUP_MEM_DIR/memory.stat" | grep 'total_mapped_file' | awk '{print $2}') bytes"
    echo "Swap: $(cat "$CGROUP_MEM_DIR/memory.stat" | grep 'total_swap' | awk '{print $2}') bytes"
    echo "Cache: $(cat "$CGROUP_MEM_DIR/memory.stat" | grep 'total_cache' | awk '{print $2}') bytes"
    echo "Hard limit: $MEM_LIMIT bytes"
    
    # 检查 OOM 状态
    OOM_STATUS=$(cat "$CGROUP_MEM_DIR/memory.oom_control")
    if [[ "$OOM_STATUS" == *"under_oom 1"* ]]; then
        echo "WARNING: Process is under OOM (Out of Memory) condition!"
    fi
    
    echo "------"
    sleep 1
done

echo "Program has finished or was terminated."