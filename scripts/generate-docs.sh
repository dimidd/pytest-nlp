#!/bin/bash
# Generate README files from templates
# This script injects code examples from test files into the documentation

set -e

cd "$(dirname "$0")/.."

echo "Copying templates..."
cp README.template.md README.md
cp MEDICAL_README.template.md MEDICAL_README.md

echo "Injecting code blocks from test files..."
npx markdown-autodocs@1.0.133 -c code-block -o ./README.md ./MEDICAL_README.md

echo "Done! README.md and MEDICAL_README.md have been updated."

