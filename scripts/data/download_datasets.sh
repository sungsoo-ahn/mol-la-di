#!/bin/bash
# Download QM9 and ZINC250k datasets from GEEL repository
# https://github.com/yunhuijang/GEEL/tree/main/resource

set -e

DATA_DIR="data/datasets"
GEEL_RAW="https://raw.githubusercontent.com/yunhuijang/GEEL/main/resource"
GEEL_MEDIA="https://media.githubusercontent.com/media/yunhuijang/GEEL/main/resource"

mkdir -p "$DATA_DIR/qm9"
mkdir -p "$DATA_DIR/zinc250k"

echo "Downloading QM9 dataset..."
for split in train val test; do
    echo "  - qm9_graph_${split}.pkl"
    curl -fL -o "$DATA_DIR/qm9/qm9_graph_${split}.pkl" \
        "$GEEL_RAW/qm9/qm9_graph_${split}.pkl"
    echo "  - qm9_smiles_${split}.txt"
    curl -fL -o "$DATA_DIR/qm9/qm9_smiles_${split}.txt" \
        "$GEEL_RAW/qm9/qm9_smiles_${split}.txt"
done

echo "Downloading ZINC250k dataset..."
# Train file is stored with Git LFS, use media CDN
echo "  - zinc_graph_train.pkl (LFS)"
curl -fL -o "$DATA_DIR/zinc250k/zinc_graph_train.pkl" \
    "$GEEL_MEDIA/zinc/zinc_graph_train.pkl"

# Other files are regular
for split in val test; do
    echo "  - zinc_graph_${split}.pkl"
    curl -fL -o "$DATA_DIR/zinc250k/zinc_graph_${split}.pkl" \
        "$GEEL_RAW/zinc/zinc_graph_${split}.pkl"
done

for split in train val test; do
    echo "  - zinc_smiles_${split}.txt"
    curl -fL -o "$DATA_DIR/zinc250k/zinc_smiles_${split}.txt" \
        "$GEEL_RAW/zinc/zinc_smiles_${split}.txt"
done

echo ""
echo "Done! Datasets downloaded to $DATA_DIR"
echo ""
echo "QM9: $(ls -1 $DATA_DIR/qm9/*.pkl 2>/dev/null | wc -l) graph files"
echo "ZINC250k: $(ls -1 $DATA_DIR/zinc250k/*.pkl 2>/dev/null | wc -l) graph files"
