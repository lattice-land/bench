#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Set the directory based on the argument
DIRECTORY=$1

# Prepare the output CSV file
echo "\"category\",\"problem\",\"model_data_file\""

# Find all .xml files in the specified directory, format them, and append to the CSV
find "$DIRECTORY" -type f -name '*.xml' | while read -r file; do
  # Extract category from the path
  category=$(echo "$file" | awk -F/ '{print $(NF-2)"_"$(NF-1)}')
  problem=$(basename "$file" ".xml")
  # Write to CSV
  echo "\"$category\",\"$problem\",\"$file\""
done
