#!/bin/bash
# Install dependencies for multilingual evaluation with translation and dual classification

echo "Installing multilingual evaluation dependencies..."

# Google Translate
pip install googletrans==4.0.0-rc1

# Ensure transformers is up to date (needed for Llama Guard 4)
pip install --upgrade transformers

echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "python scripts/evaluate_multilingual_with_translation.py \\"
echo "    --input initial_malicious_final.csv \\"
echo "    --language bengali \\"
echo "    --output outputs/evaluation_results/"
