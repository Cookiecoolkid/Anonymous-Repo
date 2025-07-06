#!/bin/bash

# ---------- 用户设置项 ----------
CGROUP_NAME="mymemgroup"
CGROUP_DIR="/sys/fs/cgroup/memory/${CGROUP_NAME}"
MEM_LIMIT=$((512 * 1024 * 1024))        # 512MB
MEM_SOFT_LIMIT=$((256 * 1024 * 1024))   # soft limit = 256MB
SWAPPINESS=0                            # 禁用 swap 建议
SEARCH_COMMAND="./run_search.sh"        # 你的原始搜索脚本
# ----------------------------------

# 创建 cgroup 并设置参数
sudo mkdir -p "$CGROUP_DIR"
echo "Setting memory limits..."
echo $MEM_LIMIT       | sudo tee "$CGROUP_DIR/memory.limit_in_bytes"
echo $MEM_SOFT_LIMIT  | sudo tee "$CGROUP_DIR/memory.soft_limit_in_bytes"
echo $SWAPPINESS      | sudo tee "$CGROUP_DIR/memory.swappiness"

# 启动搜索程序
echo "Launching $SEARCH_COMMAND under cgroup: $CGROUP_NAME"
$SEARCH_COMMAND "$@" &
PID=$!

echo $PID | sudo tee "$CGROUP_DIR/tasks"

# 实时监控内存使用
echo "Monitoring memory usage..."
while kill -0 $PID 2>/dev/null; do
    MEM_CUR=$(cat "$CGROUP_DIR/memory.usage_in_bytes")
    echo "[PID $PID] Memory Usage: $((MEM_CUR / 1024 / 1024)) MB"
    sleep 2
done

echo "Program finished or was terminated."
