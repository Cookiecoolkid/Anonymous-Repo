#!/bin/bash

CGROUP_DIR="/sys/fs/cgroup/memory/mymemgroup"
MEM_LIMIT=$((512 * 1024 * 1024))      # 最大512MB
MEM_SOFT_LIMIT=$((256 * 1024 * 1024)) # 超过256MB表示“高水位线”

# 创建 cgroup
if [ ! -d "$CGROUP_DIR" ]; then
    sudo mkdir "$CGROUP_DIR"
fi

# 设置限制
sudo sh -c "echo $MEM_LIMIT > $CGROUP_DIR/memory.limit_in_bytes"
sudo sh -c "echo $MEM_SOFT_LIMIT > $CGROUP_DIR/memory.soft_limit_in_bytes"
sudo sh -c "echo 0 > $CGROUP_DIR/memory.swappiness"

# 启动程序
./test_search_mmap "$@" &
PID=$!
echo $PID | sudo tee "$CGROUP_DIR/tasks"

# 监控内存使用
while kill -0 $PID 2>/dev/null; do
    echo "Memory usage: $(cat "$CGROUP_DIR/memory.usage_in_bytes") bytes"
    echo "Memory soft limit: $MEM_SOFT_LIMIT bytes"
    echo "Memory hard limit: $MEM_LIMIT bytes"
    
    echo "------"
    sleep 1
done

echo "Program has finished or was terminated."
