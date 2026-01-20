#!/bin/bash
# Install git hooks for this repository

set -e
cd "$(dirname "$0")/.."

echo "Installing git hooks..."
mkdir -p .git/hooks
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
echo "✅ Pre-commit hook installed"
echo ""
echo "The hook will automatically regenerate READMEs when you commit changes to:"
echo "  - tests/test_readme_examples.py"
echo "  - tests/test_medical_readme_examples.py"
echo "  - README.template.md"
echo "  - MEDICAL_README.template.md"

