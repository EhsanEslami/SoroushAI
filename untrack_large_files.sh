#!/bin/bash

# Set the size threshold in bytes (100MB)
SIZE_LIMIT=$((100 * 1024 * 1024))

# Find large files in the Git repository
LARGE_FILES=$(find . -type f -size +${SIZE_LIMIT}c -not -path "./.git/*")

# Check if any large files were found
if [ -z "$LARGE_FILES" ]; then
    echo "No files larger than 100MB found."
    exit 0
fi

# Add large files to .gitignore and untrack them
for file in $LARGE_FILES; do
    if ! grep -qxF "$file" .gitignore; then
        echo "$file" >> .gitignore
        echo "Added $file to .gitignore"
    fi
    git rm --cached "$file"
    echo "Untracked $file from Git"
done
