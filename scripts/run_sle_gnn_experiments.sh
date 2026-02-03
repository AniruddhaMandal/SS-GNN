#!/bin/bash
# SLE-GNN Experiments Runner
#
# This script runs SLE-GNN experiments on various datasets with multiple GNN types
# and compares with vanilla GNN baselines.
#
# Usage:
#   chmod +x scripts/run_sle_gnn_experiments.sh
#   ./scripts/run_sle_gnn_experiments.sh [--all|--node|--graph|--baseline]
#
# GNN Types: GCN, GIN, GAT, GraphSAGE, SGC

set -e

# Configuration
SEEDS="42 10 32 29 75"
OUTPUT_DIR="experiments/sle_gnn"
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"

# GNN types to run
GNN_TYPES=("gcn" "gin" "gat" "sage" "sgc")

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_section() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "========================================" >> "$LOG_FILE"
    echo "$1" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
}

run_experiment() {
    local config=$1
    local name=$(basename "$config" .json)

    if [ ! -f "$config" ]; then
        echo -e "${YELLOW}Skipping (not found): $config${NC}"
        return
    fi

    log "Running: $name"

    # Run with multiple seeds
    python main.py -c "$config" -m --seeds $SEEDS 2>&1 | tee -a "$LOG_FILE"

    log "Completed: $name"
    echo "----------------------------------------"
}

# ============================================================
# Node Classification Experiments
# ============================================================
run_node_classification() {
    log_section "Node Classification Experiments"

    # Datasets
    local planetoid_datasets=("cora" "citeseer" "pubmed")
    local heterophilic_datasets=("chameleon" "squirrel" "texas" "cornell" "wisconsin")

    # Run all GNN types on Planetoid datasets
    log "--- Planetoid Datasets (Cora, CiteSeer, PubMed) ---"
    for gnn in "${GNN_TYPES[@]}"; do
        log ">> GNN Type: ${gnn^^}"
        for dataset in "${planetoid_datasets[@]}"; do
            config="configs/sle_gnn/node_classification/${gnn}-${dataset}.json"
            run_experiment "$config"
        done
    done

    # Run all GNN types on Heterophilic datasets
    log "--- Heterophilic Datasets ---"
    for gnn in "${GNN_TYPES[@]}"; do
        log ">> GNN Type: ${gnn^^}"
        for dataset in "${heterophilic_datasets[@]}"; do
            config="configs/sle_gnn/node_classification/${gnn}-${dataset}.json"
            run_experiment "$config"
        done
    done
}

# ============================================================
# Graph Classification Experiments
# ============================================================
run_graph_classification() {
    log_section "Graph Classification Experiments"

    # TUDataset graphs
    local tu_datasets=("mutag" "proteins" "enzymes")

    # Run all GNN types on TUDataset
    log "--- TUDataset ---"
    for gnn in "${GNN_TYPES[@]}"; do
        log ">> GNN Type: ${gnn^^}"
        for dataset in "${tu_datasets[@]}"; do
            config="configs/sle_gnn/graph_classification/${gnn}-${dataset}.json"
            run_experiment "$config"
        done
    done
}

# ============================================================
# Run specific GNN type only
# ============================================================
run_gnn_type() {
    local gnn_type=$1
    log_section "Running ${gnn_type^^} experiments only"

    # Node classification
    log "--- Node Classification with ${gnn_type^^} ---"
    for config in configs/sle_gnn/node_classification/${gnn_type}-*.json; do
        run_experiment "$config"
    done

    # Graph classification
    log "--- Graph Classification with ${gnn_type^^} ---"
    for config in configs/sle_gnn/graph_classification/${gnn_type}-*.json; do
        run_experiment "$config"
    done
}

# ============================================================
# Baseline (Vanilla GNN) Experiments for comparison
# ============================================================
run_baselines() {
    log_section "Baseline (Vanilla GNN) Experiments"

    # Run vanilla GNN on same datasets for comparison
    for config in configs/vanilla/TUData/gcn-mutag.json; do
        if [ -f "$config" ]; then
            run_experiment "$config"
        fi
    done
}

# ============================================================
# Print help
# ============================================================
print_help() {
    echo "SLE-GNN Experiments Runner"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --all       Run all experiments (default)"
    echo "  --node      Run node classification experiments only"
    echo "  --graph     Run graph classification experiments only"
    echo "  --baseline  Run baseline experiments only"
    echo "  --gcn       Run GCN experiments only"
    echo "  --gin       Run GIN experiments only"
    echo "  --gat       Run GAT experiments only"
    echo "  --sage      Run GraphSAGE experiments only"
    echo "  --sgc       Run SGC experiments only"
    echo "  --help      Show this help message"
    echo ""
    echo "GNN Types supported: GCN, GIN, GAT, GraphSAGE, SGC"
    echo ""
    echo "Examples:"
    echo "  $0 --all           # Run all experiments"
    echo "  $0 --node          # Run node classification only"
    echo "  $0 --gat           # Run GAT experiments only"
    echo "  $0 --gcn --node    # Run GCN node classification only"
}

# ============================================================
# Main
# ============================================================
main() {
    log "Starting SLE-GNN Experiments"
    log "Seeds: $SEEDS"
    log "Output directory: $OUTPUT_DIR"
    log "GNN Types: ${GNN_TYPES[*]}"

    case "${1:-all}" in
        --help|-h)
            print_help
            exit 0
            ;;
        --node)
            run_node_classification
            ;;
        --graph)
            run_graph_classification
            ;;
        --baseline)
            run_baselines
            ;;
        --gcn)
            run_gnn_type "gcn"
            ;;
        --gin)
            run_gnn_type "gin"
            ;;
        --gat)
            run_gnn_type "gat"
            ;;
        --sage)
            run_gnn_type "sage"
            ;;
        --sgc)
            run_gnn_type "sgc"
            ;;
        --all|*)
            run_node_classification
            run_graph_classification
            run_baselines
            ;;
    esac

    log "All experiments completed!"
    log "Results saved to: $OUTPUT_DIR"
}

main "$@"
