#!/bin/bash

#######################################
# GNN Experiment Runner
# Usage: ./run_experiments.sh
#######################################

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment_logs/${TIMESTAMP}"
RESULTS_DIR="results/${TIMESTAMP}"
CONFIG_DIR="config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Log file for summary
SUMMARY_LOG="${LOG_DIR}/summary.txt"

# List of config files to run
CONFIGS=(
    "configs/vanilla/LRGB/PCQM-Contact/gcn-pcqm-contact.json"
    "configs/vanilla/LRGB/PCQM-Contact/gine-pcqm-contact.json"
    "configs/vanilla/TUData/gcn-collab.json"
    # Add more config files here
)

# Initialize counters
total_experiments=${#CONFIGS[@]}
completed=0
failed=0
start_time=$(date +%s)

# Function to format duration
format_duration() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Function to send notification (optional - requires notify-send or similar)
send_notification() {
    local title="$1"
    local message="$2"
    if command -v notify-send &> /dev/null; then
        notify-send "$title" "$message"
    fi
}

# Print header
echo "=========================================" | tee "${SUMMARY_LOG}"
echo "GNN Experiment Runner" | tee -a "${SUMMARY_LOG}"
echo "Started: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Total experiments: ${total_experiments}" | tee -a "${SUMMARY_LOG}"
echo "=========================================" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# Main experiment loop
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    exp_num=$((i + 1))
    exp_name=$(basename "$config" .json)
    
    echo -e "${BLUE}[${exp_num}/${total_experiments}] Starting: ${exp_name}${NC}"
    echo "[${exp_num}/${total_experiments}] Starting: ${exp_name}" >> "${SUMMARY_LOG}"
    echo "  Config: ${config}" | tee -a "${SUMMARY_LOG}"
    echo "  Time: $(date)" | tee -a "${SUMMARY_LOG}"
    
    # Check if config file exists
    if [ ! -f "$config" ]; then
        echo -e "${RED}  ✗ Config file not found: ${config}${NC}" | tee -a "${SUMMARY_LOG}"
        ((failed++))
        echo "" | tee -a "${SUMMARY_LOG}"
        continue
    fi
    
    # Run experiment
    exp_start=$(date +%s)
    log_file="${LOG_DIR}/${exp_name}.log"
    
    # Option 1: Filter progress bar lines (recommended)
    gps-run -c "$config" 2>&1 | grep -v -e "^[[:space:]]*[0-9]*%|" -e "it/s" -e "s/it" > "$log_file"
    exit_code=${PIPESTATUS[0]}
    
    # Option 2: Use unbuffer to get clean output (uncomment if Option 1 doesn't work)
    # Requires: sudo apt install expect-dev
    # unbuffer gps-run -c "$config" 2>&1 | tee "$log_file" | grep --line-buffered -v -e "^[[:space:]]*[0-9]*%|" -e "it/s" > /dev/null
    # exit_code=${PIPESTATUS[0]}
    
    exp_end=$(date +%s)
    duration=$((exp_end - exp_start))
    
    # Check result
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✓ Completed successfully${NC}" | tee -a "${SUMMARY_LOG}"
        echo "  Duration: $(format_duration $duration)" | tee -a "${SUMMARY_LOG}"
        ((completed++))
    else
        echo -e "${RED}  ✗ Failed (exit code: ${exit_code})${NC}" | tee -a "${SUMMARY_LOG}"
        echo "  Duration: $(format_duration $duration)" | tee -a "${SUMMARY_LOG}"
        echo "  Check log: ${log_file}" | tee -a "${SUMMARY_LOG}"
        ((failed++))
    fi
    
    # Show progress
    remaining=$((total_experiments - exp_num))
    echo "  Progress: ${exp_num}/${total_experiments} | Remaining: ${remaining}" | tee -a "${SUMMARY_LOG}"
    echo "" | tee -a "${SUMMARY_LOG}"
    
    # Optional: Add delay between experiments (uncomment if needed)
    # sleep 5
done

# Print summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "=========================================" | tee -a "${SUMMARY_LOG}"
echo "Experiment Run Complete" | tee -a "${SUMMARY_LOG}"
echo "=========================================" | tee -a "${SUMMARY_LOG}"
echo "Finished: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Total duration: $(format_duration $total_duration)" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"
echo -e "${GREEN}Completed: ${completed}${NC}" | tee -a "${SUMMARY_LOG}"
echo -e "${RED}Failed: ${failed}${NC}" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"
echo "Logs saved to: ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
echo "Results saved to: ${RESULTS_DIR}" | tee -a "${SUMMARY_LOG}"
echo "=========================================" | tee -a "${SUMMARY_LOG}"

# Send completion notification
send_notification "GNN Experiments Complete" "Completed: ${completed}, Failed: ${failed}"

# Exit with appropriate code
if [ $failed -gt 0 ]; then
    exit 1
else
    exit 0
fi