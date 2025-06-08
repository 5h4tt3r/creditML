# Credit Card Fraud Detection

## Project Overview
Binary classification project for detecting fraudulent credit card transactions.

## Team Members
- Gurick Kooner
- Kaori Kamata
- Gaunghan Tong

## Setup Instructions
1. Clone repository = git pull 
2. Download and setup Conda for terminal
3. Create conda environment: `conda create -n fraud-detect python=3.10`
4. Activate environment: `conda activate fraud-detect
5. Install dependencies: `pip install -r requirements.txt`

## Github Daily Workflow in Terminal
# Start work
git checkout develop (your feature branch)
git pull origin develop
git checkout -b feature/(yourbranch)

# Work on notebooks/code
# Commit frequently with descriptive messages
git add .
git commit -m "Add initial EDA for transaction patterns"

# Push and create pull request
git push origin feature/john-eda

## Project Structure
- `data/` - Dataset storage
- `notebooks/` - Jupyter notebooks
- `src/` - Source code modules
- `models/` - Saved model files
