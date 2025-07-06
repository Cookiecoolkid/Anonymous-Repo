#!/bin/bash

CGROUP_DIR="/sys/fs/cgroup/memory/mymemgroup"
BLKIO_DIR="/sys/fs/cgroup/blkio/mymemgroup"

MEM_LIMIT=$((512 * 1024 * 1024))      # 最大512MB
MEM_SOFT_LIMIT=$((256 * 1024 * 1024)) # 高水位线

# 创建 cgroup（memory 和 blkio）
for dir in "$CGROUP_DIR" "$BLKIO_DIR"; do
    if [ ! -d "$dir" ]; then
        sudo mkdir -p "$dir"
    fi
done

# 设置 memory 限制
sudo sh -c "echo $MEM_LIMIT > $CGROUP_DIR/memory.limit_in_bytes"
sudo sh -c "echo $MEM_SOFT_LIMIT > $CGROUP_DIR/memory.soft_limit_in_bytes"
sudo sh -c "echo 0 > $CGROUP_DIR/memory.swappiness"

# 启动程序
./test_search_mmap "$@" &
PID=$!
echo $PID | sudo tee "$CGROUP_DIR/tasks"
echo $PID | sudo tee "$BLKIO_DIR/tasks"

# 监控内存与IO
while kill -0 $PID 2>/dev/null; do
    echo "[Memory]"
    echo "Usage: $(cat "$CGROUP_DIR/memory.usage_in_bytes") bytes"
    echo "Soft limit: $MEM_SOFT_LIMIT bytes"
    echo "Hard limit: $MEM_LIMIT bytes"

    echo "[Block IO]"
    cat "$BLKIO_DIR/blkio.io_serviced"
    cat "$BLKIO_DIR/blkio.io_service_bytes"
    
    echo "------"
    sleep 1
done

echo "Program has finished or was terminated."
