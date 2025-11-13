#!/bin/ash

# Function to run a script and log its output
run_script() {
    local script=$1
    local log_file=$2
    $script 2>&1 | tee -a $log_file
}

# Log files
log_file_1="/var/log/zenoh_bridge_create3.log"
log_file_2="/var/log/zenoh_bridge_rpi.log"

# Ensure the log directory exists
mkdir -p /var/log

# Execute the first script and log its output
run_script /app/run_zenoh_bridge_create3.sh $log_file_1 &

# Get PID of the first script
pid1=$!

# Execute the second script and log its output
run_script /app/run_zenoh_bridge_rpi.sh $log_file_2 &

# Get PID of the second script
pid2=$!

# Wait for both scripts to finish
wait $pid1
wait $pid2
