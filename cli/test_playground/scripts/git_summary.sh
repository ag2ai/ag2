#!/bin/bash
# Summarize the current git repository
echo "=== Git Summary ==="
echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a repo')"
echo "Last commit: $(git log -1 --oneline 2>/dev/null || echo 'N/A')"
echo "Status:"
git status --short 2>/dev/null || echo "Not a git repository"
