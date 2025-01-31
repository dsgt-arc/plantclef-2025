#!/usr/bin/env bash

# This script extracts dataset files into specific directories.
# - TAR files are extracted into data/images/train/, data/images/test/, and data/models/.
# - CSV files are moved into data/metadata/.

# Usage:
# scripts/extract_dataset.sh [SOURCE_DIR]
#
# Example:
# scripts/extract_dataset.sh ~/scratch/data/raw
#
# This will extract tar files into their respective directories and move the CSV file.
# To run this script, run the following commands:
# chmod +x extract-data.sh
# ./extract-data.sh

set -e  # Exit immediately if a command exits with a non-zero status.

# Default source directory
SOURCE_DIR=${1:-"$HOME/scratch/data/"}
DEST_DIR=${1:-"$HOME/p-dsgt_clef2025-0/shared/plantclef/data"}
TRAIN_DIR="$DEST_DIR/train"
TEST_DIR="$DEST_DIR/test"
METADATA_DIR="$DEST_DIR/metadata"
MODELS_DIR="$DEST_DIR/models"

# Ensure directories exist
for dir in "$TRAIN_DIR" "$TEST_DIR" "$METADATA_DIR" "$MODELS_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

echo "Extracting datasets from $SOURCE_DIR..."

# Extract dataset tar files
for file in "$SOURCE_DIR"/raw/*.tar; do
    if [[ "$file" == *"2024singleplanttrainingdata.tar" ]]; then
        echo "Extracting $file into $TRAIN_DIR"
        tar -xf "$file" -C "$TRAIN_DIR"
    elif [[ "$file" == *"2025test.tar" ]]; then
        echo "Extracting $file into $TEST_DIR"
        tar -xf "$file" -C "$TEST_DIR"
    fi
done

# Extract model tar files
for file in "$SOURCE_DIR"/models/*.tar; do
    if [[ "$file" == *"PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar" ]]; then
        echo "Extracting $file into $MODELS_DIR"
        tar -xf "$file" -C "$MODELS_DIR"
    fi
done

# Move CSV files
for file in "$SOURCE_DIR"/raw/*.csv; do
    echo "Moving $file to $METADATA_DIR"
    mv "$file" "$METADATA_DIR/"
done

# Final listing
echo "Final contents of $DEST_DIR:"
ls -R "$DEST_DIR"

echo "Extraction completed successfully."
