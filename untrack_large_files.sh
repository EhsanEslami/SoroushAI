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
    # Convert ./file to file (remove leading ./)
    REL_PATH=$(realpath --relative-to="$(git rev-parse --show-toplevel)" "$file")

    if ! grep -qxF "$REL_PATH" .gitignore; then
        echo "$REL_PATH" >> .gitignore
        echo "Added $REL_PATH to .gitignore"
    fi
    git rm --cached "$REL_PATH"
    echo "Untracked $REL_PATH from Git"
done
