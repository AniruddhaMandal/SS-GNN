#!/bin/bash
# Phase 1: Run all datasets for SD-GNN vs Vanilla comparison
# Usage: ./scripts/run_phase1_overnight.sh

set -e  # Exit on error

echo "=========================================="
echo "Phase 1: SD-GNN vs Vanilla Experiments"
echo "Started at: $(date)"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Results file
RESULTS_FILE="experiment_results/phase1_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p experiment_results

echo "Results will be saved to: $RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Phase 1 Experiments - $(date)" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"

# Function to run experiment and log results
run_experiment() {
    local config=$1
    local name=$2
    echo ""
    echo ">>> Running: $name"
    echo "    Config: $config"
    echo "-------------------------------------------" >> "$RESULTS_FILE"
    echo "$name" >> "$RESULTS_FILE"

    if gps-run -c "$config" -p 2>&1 | tee -a "$RESULTS_FILE"; then
        echo "    ✓ Completed"
    else
        echo "    ✗ Failed"
        echo "FAILED" >> "$RESULTS_FILE"
    fi
}

echo ""
echo "=== TUDataset Experiments ==="

# MUTAG (already have results, but run for completeness)
run_experiment "configs/ss_gnn/TUData/gcn-mutag.json" "SD-GNN: MUTAG"
run_experiment "configs/vanilla/TUData/gcn-mutag.json" "Vanilla: MUTAG"

# PTC_MR
run_experiment "configs/ss_gnn/TUData/gcn-ptc_mr.json" "SD-GNN: PTC_MR"
run_experiment "configs/vanilla/TUData/gcn-ptc_mr.json" "Vanilla: PTC_MR"

# AIDS
run_experiment "configs/ss_gnn/TUData/gcn-aids.json" "SD-GNN: AIDS"
run_experiment "configs/vanilla/TUData/gcn-aids.json" "Vanilla: AIDS"

# PROTEINS
run_experiment "configs/ss_gnn/TUData/gcn-proteins.json" "SD-GNN: PROTEINS"
run_experiment "configs/vanilla/TUData/gcn-proteins.json" "Vanilla: PROTEINS"

# IMDB-BINARY
run_experiment "configs/ss_gnn/TUData/gcn-imdb-binary.json" "SD-GNN: IMDB-BINARY"
run_experiment "configs/vanilla/TUData/gcn-imdb-binary.json" "Vanilla: IMDB-BINARY"

echo ""
echo "=== MoleculeNet Experiments ==="

# BBBP
run_experiment "configs/ss_gnn/MoleculeNet/gcn-bbbp.json" "SD-GNN: BBBP"
run_experiment "configs/vanilla/MoleculeNet/gcn-bbbp.json" "Vanilla: BBBP"

# Tox21
run_experiment "configs/ss_gnn/MoleculeNet/gcn-tox21.json" "SD-GNN: Tox21"
run_experiment "configs/vanilla/MoleculeNet/gcn-tox21.json" "Vanilla: Tox21"

echo ""
echo "=== Synthetic Experiments ==="

# CSL
run_experiment "configs/ss_gnn/SYNTHETIC/gcn-csl.json" "SD-GNN: CSL"
run_experiment "configs/vanilla/SYNTHETIC/CSL/gcn-csl.json" "Vanilla: CSL"

echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
