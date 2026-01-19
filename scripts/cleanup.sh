#!/bin/bash
# Clean up experiment outputs while preserving datasets

rm -rf data/runs/*
touch data/runs/.gitkeep

echo "Cleaned experiment outputs in data/runs/"
echo "Datasets in data/datasets/ preserved."
