#!/bin/bash
# Report disk usage for a given path (default: current directory)
PATH_ARG="${1:-.}"
echo "Disk usage for: $PATH_ARG"
du -sh "$PATH_ARG" 2>/dev/null || echo "Error: cannot read $PATH_ARG"
echo ""
echo "Top 5 largest items:"
du -sh "$PATH_ARG"/* 2>/dev/null | sort -rh | head -5
