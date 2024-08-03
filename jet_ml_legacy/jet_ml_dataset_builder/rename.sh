#!/bin/bash

# Loop through all items in the current directory
for item in *; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        # Replace hyphens and dots with underscores in directory name
        new_name=$(echo "$item" | sed 's/[-.]/_/g')
        # Rename the directory
        mv "$item" "$new_name"
        echo "Renamed directory: $item to $new_name"
    elif [ -f "$item" ]; then
        # Extract file extension
        extension="${item##*.}"
        # Replace hyphens and dots with underscores in file name
        file_name="${item%.*}"
        new_file_name=$(echo "$file_name" | sed 's/[-.]/_/g')
        # Rename the file with extension
        mv "$item" "${new_file_name}.${extension}"
        echo "Renamed file: $item to ${new_file_name}.${extension}"
    fi
done
