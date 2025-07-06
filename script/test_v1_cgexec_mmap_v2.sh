#!/bin/bash

#=============================#
#       CONFIGURATIONS       #
#=============================#
CGROUP_NAME="mymem"
MEM_LIMIT_MB=512
MEM_LIMIT=$((MEM_LIMIT_MB * 1024 * 1024))
CGROUP_PATH="/sys/fs/cgroup/memory/$CGROUP_NAME"
MONITOR_INTERVAL=1
LOGFILE="$HOME/program_output.log"
TARGET_EXECUTABLE_NAME="$(basename "$1")"  # e.g., test_search_mmap

#=============================#
#       CLEANUP CGROUP        #
#=============================#
cleanup() {
    echo "Cleaning up..."
    sudo cgdelete -g memory:$CGROUP_NAME 2>/dev/null || true
}
cleanup

#=============================#
#     SETUP CGROUP + LIMIT   #
#=============================#
sudo cgcreate -g memory:$CGROUP_NAME || { echo "❌ Failed to create cgroup"; exit 1; }
sudo cgset -r memory.limit_in_bytes=$MEM_LIMIT $CGROUP_NAME

#=============================#
#     RUN TARGET IN CGROUP   #
#=============================#
echo "Launching program in cgroup $CGROUP_NAME with ${MEM_LIMIT_MB}MB limit..."
sudo cgexec -g memory:$CGROUP_NAME "$@" > >(tee "$LOGFILE") 2>&1 &
SUDO_PID=$!

#=============================#
#     WAIT FOR TARGET PID    #
#=============================#
echo "Waiting for actual $TARGET_EXECUTABLE_NAME to spawn..."
TARGET_NAME=$(basename "$1")
echo "base name is $TARGET_NAME"
TARGET_PID=""

for i in {1..10}; do
    TARGET_PID=$(ps -eo pid,args | awk -v target="$TARGET_NAME" 'index($0, target) { print $1; exit }')
    if [ -n "$TARGET_PID" ]; then
        break
    fi
    sleep 0.5
done

#=============================#
#   CHECK AND ASSIGN CGROUP  #
#=============================#
if [ -z "$TARGET_PID" ]; then
    echo "❌ Failed to find real $TARGET_EXECUTABLE_NAME process after waiting"
    kill $SUDO_PID 2>/dev/null
    exit 1
fi

# 强制加入 CGROUP（防止 cgexec 异常）
echo "$TARGET_PID" | sudo tee "$CGROUP_PATH/tasks" > /dev/null
echo "✅ Added PID $TARGET_PID to cgroup $CGROUP_NAME"

#=============================#
#       MEMORY MONITORING    #
#=============================#
PGID=$(ps -o pgid= $TARGET_PID | tr -d ' ')
CMD=$(ps -p $TARGET_PID -o cmd=)
echo "Target PID: $TARGET_PID (CMD: $CMD), PGID: $PGID"

monitor_memory() {
    echo "📈 Monitoring memory usage..."
    while ps -p $TARGET_PID >/dev/null 2>&1; do
        USAGE=$(cat "$CGROUP_PATH/memory.usage_in_bytes" 2>/dev/null || echo 0)
        MAX_USAGE=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes" 2>/dev/null || echo 0)
        echo "[$(date +%T)] Current: $((USAGE / 1024 / 1024))MB | Peak: $((MAX_USAGE / 1024 / 1024))MB / ${MEM_LIMIT_MB}MB"
        sleep $MONITOR_INTERVAL
    done
}
monitor_memory &

#=============================#
#          WAIT EXIT         #
#=============================#
wait $SUDO_PID
kill $! 2>/dev/null

#=============================#
#         OOM CHECK          #
#=============================#
if grep -q "oom_kill 1" "$CGROUP_PATH/memory.oom_control"; then
    echo "❌ OOM killer triggered"
    dmesg | grep -i 'killed process' | tail -n 5
fi

FINAL_PEAK=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes")
echo "Final peak memory usage: $((FINAL_PEAK / 1024 / 1024))MB"

cleanup
echo "✅ Program complete."
