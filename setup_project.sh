#!/bin/bash

# Setup script for SEC RAG System

echo "Setting up SEC RAG System..."

# Create directory structure
echo "Creating directories..."
mkdir -p src
mkdir -p data/raw
mkdir -p models
mkdir -p tests

# Create empty __init__.py
echo "Creating __init__.py..."
touch src/__init__.py

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch models/.gitkeep

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

echo ""
echo "✅ Project structure created!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Create .env file with your OPENAI_API_KEY"
echo "4. Place apple_data.pdf in data/raw/"
echo "5. Run test: python test_rag_system.py"