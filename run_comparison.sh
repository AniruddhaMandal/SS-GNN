#!/bin/bash
# ==============================================================================
# Run SLE-GNN vs VANILLA comparison on node classification benchmarks
#
# Usage:
#   ./run_comparison.sh                    # default: gcn
#   ./run_comparison.sh --conv_type gin    # use GIN convolution
#   ./run_comparison.sh --conv_type sage   # use GraphSAGE
#   ./run_comparison.sh --conv_type sgc    # use SGC (Simplified Graph Convolution)
#   ./run_comparison.sh --conv_type gcnii  # use GCNII
#   ./run_comparison.sh --conv_type pna    # use PNA (Principal Neighbourhood Aggregation)
#   ./run_comparison.sh --conv_type jknet  # use JKNet (GCN + Jumping Knowledge)
#   ./run_comparison.sh --conv_type gcn --multi_seed  # multi-seed runs
# ==============================================================================

set -e

# ---------- Defaults ----------
CONV_TYPE="gcn"
MULTI_SEED=""
SEEDS="42 10 32 29 75"

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conv_type)
            CONV_TYPE="$2"
            shift 2
            ;;
        --multi_seed)
            MULTI_SEED="--multi-seed"
            shift
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--conv_type gcn|gin|sage|gat|gatv2|sgc|gcnii|pna|jknet] [--multi_seed] [--seeds '42 10 32']"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo " SLE-GNN vs VANILLA Comparison"
echo " Conv type: ${CONV_TYPE}"
echo " Residual:  OFF"
echo " Multi-seed: ${MULTI_SEED:-no}"
echo "=============================================="

# ---------- Dataset configs ----------
# Format: "config_file_suffix:model_dir"
# Config file suffix maps to gcn-{suffix}.json in both vanilla/ and sle_gnn/ dirs

DATASETS=(
    "cora"
    "citeseer"
    "pubmed"
    "amazon-photo"
    "coauthor-cs"
    "coauthor-physics"
    "amazon-computers"
    "chameleon"
    "squirrel"
    "actor"
    "wisconsin"
    "texas"
    "cornell"
    "roman-empire"
    "amazon-ratings"
    "minesweeper"
    "tolokers"
    "questions"
)

VANILLA_DIR="configs/vanilla/node_classification"
SLE_DIR="configs/sle_gnn/node_classification"

# Common overrides: force conv type and residual=false
OVERRIDES="-o model_config.mpnn_type=${CONV_TYPE} model_config.kwargs.residual=false"

# ---------- Run experiments ----------
TOTAL=${#DATASETS[@]}
COUNT=0

for ds in "${DATASETS[@]}"; do
    COUNT=$((COUNT + 1))
    CONFIG_FILE="gcn-${ds}.json"

    echo ""
    echo "======================================================"
    echo " [${COUNT}/${TOTAL}] Dataset: ${ds}"
    echo "======================================================"

    # --- VANILLA ---
    VANILLA_CFG="${VANILLA_DIR}/${CONFIG_FILE}"
    if [[ -f "${VANILLA_CFG}" ]]; then
        echo "  >> Running VANILLA (${CONV_TYPE}, no residual) on ${ds}..."
        python main.py -c "${VANILLA_CFG}" ${OVERRIDES} ${MULTI_SEED} ${MULTI_SEED:+--seeds ${SEEDS}}
    else
        echo "  !! VANILLA config not found: ${VANILLA_CFG}, skipping."
    fi

    # --- SLE-GNN ---
    SLE_CFG="${SLE_DIR}/${CONFIG_FILE}"
    if [[ -f "${SLE_CFG}" ]]; then
        echo "  >> Running SLE-GNN (${CONV_TYPE}, no residual) on ${ds}..."
        python main.py -c "${SLE_CFG}" ${OVERRIDES} ${MULTI_SEED} ${MULTI_SEED:+--seeds ${SEEDS}}
    else
        echo "  !! SLE-GNN config not found: ${SLE_CFG}, skipping."
    fi
done

echo ""
echo "=============================================="
echo " All experiments completed!"
echo " Results saved in experiments/ and experiment_results/"
echo "=============================================="
