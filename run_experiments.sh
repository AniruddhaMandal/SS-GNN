#!/bin/bash

#######################################
# GNN Experiment Runner with Overrides
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Log file for summary
SUMMARY_LOG="${LOG_DIR}/summary.txt"

# Define experiments with config and optional overrides
# Format: "config_path|override1 override2 override3|experiment_name"
# If no overrides needed, use: "config_path||experiment_name"
EXPERIMENTS=(
    # Examples:
    "configs/ss_gnn/TUData/gin-proteins.json|model_config.dropout=0.4"
    "configs/ss_gnn/TUData/gin-proteins.json|model_config.temperature=5.0"
    "configs/ss_gnn/TUData/gin-proteins.json|model_config.temperature=0.1"
    "configs/ss_gnn/TUData/gin-proteins.json|model_config.subgraph_param.k=12 model_config.subgraph_param.m=50"
    "configs/ss_gnn/TUData/gin-proteins.json|model_config.subgraph_param.k=6"
    
    # Multi-seed examples (add more as needed):
    # "configs/vanilla/ZINC/gine.json|--multi-seed --seeds 42 123 456|vanilla_gine_multiseed"
)

# Initialize counters
total_experiments=${#EXPERIMENTS[@]}
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

# Function to parse experiment definition
parse_experiment() {
    local exp_def="$1"
    IFS='|' read -r config overrides exp_name <<< "$exp_def"
    
    # Return values via global variables
    EXP_CONFIG="$config"
    EXP_OVERRIDES="$overrides"
    EXP_NAME="${exp_name:-$(basename "$config" .json)}"
}

# Print header
echo "=========================================" | tee "${SUMMARY_LOG}"
echo "GNN Experiment Runner" | tee -a "${SUMMARY_LOG}"
echo "Started: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Total experiments: ${total_experiments}" | tee -a "${SUMMARY_LOG}"
echo "=========================================" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# Main experiment loop
for i in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))
    
    # Parse experiment definition
    parse_experiment "$experiment"
    config="$EXP_CONFIG"
    overrides="$EXP_OVERRIDES"
    exp_name="$EXP_NAME"
    
    echo -e "${BLUE}[${exp_num}/${total_experiments}] Starting: ${exp_name}${NC}"
    echo "[${exp_num}/${total_experiments}] Starting: ${exp_name}" >> "${SUMMARY_LOG}"
    echo "  Config: ${config}" | tee -a "${SUMMARY_LOG}"
    
    # Display overrides if present
    if [ -n "$overrides" ]; then
        echo -e "  ${CYAN}Overrides: ${overrides}${NC}" | tee -a "${SUMMARY_LOG}"
    fi
    
    echo "  Time: $(date)" | tee -a "${SUMMARY_LOG}"
    
    # Check if config file exists
    if [ ! -f "$config" ]; then
        echo -e "${RED}  ✗ Config file not found: ${config}${NC}" | tee -a "${SUMMARY_LOG}"
        ((failed++))
        echo "" | tee -a "${SUMMARY_LOG}"
        continue
    fi
    
    # Build command
    cmd="gps-run -c \"$config\""
    
    # Add overrides if present
    if [ -n "$overrides" ]; then
        # Check if overrides contain multi-seed flags
        if [[ "$overrides" == *"--multi-seed"* ]] || [[ "$overrides" == *"-m"* ]]; then
            # Multi-seed case: append overrides as-is
            cmd="$cmd $overrides"
        else
            # Regular overrides: use -o flag
            cmd="$cmd -o $overrides"
        fi
    fi
    
    # Run experiment
    exp_start=$(date +%s)
    log_file="${LOG_DIR}/${exp_name}.log"
    
    # Execute command and capture output
    echo "  Command: $cmd" | tee -a "${SUMMARY_LOG}"
    
    # Option 1: Filter progress bar lines (recommended)
    eval "$cmd" 2>&1 | grep -v -e "^[[:space:]]*[0-9]*%|" -e "it/s" -e "s/it" > "$log_file"
    exit_code=${PIPESTATUS[0]}
    
    # Option 2: Use unbuffer to get clean output (uncomment if Option 1 doesn't work)
    # Requires: sudo apt install expect-dev
    # eval "unbuffer $cmd" 2>&1 | tee "$log_file" | grep --line-buffered -v -e "^[[:space:]]*[0-9]*%|" -e "it/s" > /dev/null
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