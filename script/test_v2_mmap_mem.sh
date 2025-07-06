#!/bin/bash

# 设置 cgroup 名称
CGROUP_NAME="graph"
CGROUP_DIR="/sys/fs/cgroup/$CGROUP_NAME"

# 内存限制（单位：字节）
MEM_HIGH=$((128 * 1024 * 1024))    # 512MB soft limit
MEM_MAX=$((256 * 1024 * 1024))    # 1GB hard limit

# 删除旧 cgroup（如果空则成功）
if [ -d "$CGROUP_DIR" ]; then
    echo "Cleaning up old cgroup: $CGROUP_DIR"
    sudo rmdir "$CGROUP_DIR" 2>/dev/null || true
fi

# 创建新的 cgroup v2 子组
sudo mkdir -p "$CGROUP_DIR"

# 设置内存限制
echo "$MEM_HIGH" | sudo tee "$CGROUP_DIR/memory.high" > /dev/null
echo "$MEM_MAX"  | sudo tee "$CGROUP_DIR/memory.max" > /dev/null
echo 0           | sudo tee "$CGROUP_DIR/memory.min"  > /dev/null
echo max         | sudo tee "$CGROUP_DIR/memory.swap.max" > /dev/null

# 使用 cgexec 启动程序（使用 memory 子系统）
# "$@" 表示将脚本后续参数传给测试程序
cgexec -g memory:$CGROUP_NAME "$@" &
PID=$!

echo "Monitoring PID $PID in cgroup v2: $CGROUP_DIR"
echo "Target: keep memory usage near 512MB"

# 实时监控 memory.current
while kill -0 $PID 2>/dev/null; do
    MEM_CURRENT=$(cat "$CGROUP_DIR/memory.current")
    echo "[memory.current] $MEM_CURRENT bytes"
    echo "------"
    sleep 1
done

echo "Process $PID exited."
