# What is SEC 10-k filing report?
A 10-K filing is an annual report required by the U.S. Securities and Exchange Commission (SEC) that public companies must file. It provides a comprehensive overview of the company's business, financial condition, and performance over the past fiscal year, including audited financial statements. The 10-K is the most comprehensive periodic report filed with the SEC and contains detailed information on topics such as the company's business, risks, executive compensation, and corporate governance.

# SEC 10-K RAG System - Setup Guide

## Quick Start

### 1. Project Structure
```
sec_rag_system/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── text_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── main.py
├── data/
│   └── raw/
│       └── apple_data.pdf
├── models/
│   └── .gitkeep
├── tests/
│   └── test_vector.py
│   └── api_check.py
├── config.yaml
├── requirements.txt
├── test_rag_system.py
├── .gitignore
└── README.md
```

### 2. Installation Steps

#### Step 1: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Set up OpenAI API Key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

#### Step 4: Create Directory Structure
```bash
mkdir -p data/raw models
```

#### Step 5: Add Your PDF
Place `apple_data.pdf` in the `data/raw/` directory.

### 3. Testing the System

#### Test Core Components (Optional)
```bash
python test_vector.py
```

#### Test Complete RAG Pipeline
```bash
python test_rag_system.py
```

This will:
1. Extract text from the PDF
2. Create embeddings
3. Build vector index
4. Ask 10 test questions
5. Generate answers using GPT

### 4. Using the Streamlit App (Optional)
```bash
streamlit run app.py
```

### 5. Git Setup

#### Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit: SEC 10-K RAG system"
```

#### Create GitHub Repository
1. Go to GitHub and create a new repository
2. Don't initialize with README (we have one)
3. Follow GitHub's instructions:

```bash
git remote add origin https://github.com/yourusername/sec-rag-system.git
git branch -M main
git push -u origin main
```

### 6. Expected Output

When you run `test_rag_system.py`, you should see:

```
================================================================================
TESTING COMPLETE RAG PIPELINE
================================================================================

1. Initializing RAG System...
2. Checking for existing index...
No existing index found. Processing PDF...

3. Processing the 10-K filing...
Extracting text from PDF...
Successfully extracted text from 238 pages
Total text length: 523847 characters
Cleaning extracted text...
Text cleaned. Final length: 521234 characters
Identifying sections...
Found section 'business' at position 12453
Found section 'risk_factors' at position 45678
...
Creating embeddings for 423 chunks...
Building vector store...
Vector store built successfully!

================================================================================
TESTING QUESTION ANSWERING
================================================================================

--------------------------------------------------------------------------------
QUESTION 1: What are the main risk factors mentioned in the report?
--------------------------------------------------------------------------------
Question: What are the main risk factors mentioned in the report?
Retrieving relevant context...
Generating answer...

ANSWER:
Based on the 10-K filing, Apple faces several main risk factors:

1. **Macroeconomic Conditions**: Global and regional economic conditions significantly impact...
[detailed answer continues]

--------------------------------------------------------------------------------
QUESTION 2: What is Apple's business strategy?
--------------------------------------------------------------------------------
...
```

### 7. What's Happening Under the Hood

1. **Text Extraction** (`data_loader.py`): 
   - Reads PDF file
   - Extracts text from all pages
   - Cleans the text

2. **Text Processing** (`text_processor.py`):
   - Identifies 10-K sections (Business, Risk Factors, etc.)
   - Splits text into chunks with overlap
   - Creates embeddings using sentence-transformers

3. **Vector Storage** (`vector_store.py`):
   - Stores embeddings in FAISS index
   - Enables fast similarity search
   - Saves/loads index to/from disk

4. **Retrieval** (`retriever.py`):
   - Converts question to embedding
   - Finds most similar chunks
   - Formats context for LLM

5. **Generation** (`generator.py`):
   - Sends question + context to OpenAI
   - Gets natural language answer
   - Returns formatted response