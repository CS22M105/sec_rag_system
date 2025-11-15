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
Loading existing vector store...
✅ Loaded index with 1858 chunks
Loading embedding model: all-MiniLM-L6-v2
✅ Loaded existing vector store
Existing index loaded successfully!

================================================================================
SAMPLE QUESTIONS YOU CAN ASK:
1. What are the main risk factors mentioned in the report?
2. What is Apple's business strategy?
3. What products does Apple sell?
4. What were the total revenues for the last fiscal year?
5. What segments does Apple operate in?
6. What are the major legal proceedings mentioned?
7. What is Apple's approach to research and development?
8. How does Apple manage competition?
9. What are the key markets for Apple?
10. What is Apple's dividend policy?
================================================================================

Enter your question: What are the major legal proceedings mentioned?

================================================================================
Question: What are the major legal proceedings mentioned?
================================================================================
Retrieving relevant context...
Found 5 relevant chunks
  Chunk 1: Section=financials, Score=0.517
  Chunk 2: Section=financials, Score=0.491
  Chunk 3: Section=financials, Score=0.478
  Chunk 4: Section=financials, Score=0.477
  Chunk 5: Section=financials, Score=0.474

Generating answer...
ANSWER:
The major legal proceedings mentioned in the context include:

1. **Digital Markets Act (DMA) Investigations**: On March 25, 2024, the European Commission announced two formal noncompliance investigations against the Company under the DMA. These investigations concern:
   - Article 5(4) related to how developers may communicate and promote offers to end users for apps distributed through the App Store.
   - Article 6(3) concerning default settings, uninstallation of apps, and a web browser choice screen on iOS. 

   On June 24, 2024, the Commission announced preliminary findings alleging that the Company’s App Store rules breach the DMA and opened a third investigation regarding new contractual requirements for third-party app developers.

2. **Department of Justice (DOJ) Lawsuit**: On March 21, 2024, the DOJ, along with several state and district attorneys general, filed a civil antitrust lawsuit in the U.S. District Court for the District of New Jersey against the Company. The lawsuit alleges monopolization or attempted monopolization in the markets for "performance smartphones" and "smartphones," violating U.S. antitrust laws. The DOJ seeks equitable relief to address the alleged anticompetitive behavior.

3. **Epic Games Lawsuit**: Epic Games, Inc. filed a lawsuit in the U.S. District Court for the Northern District of California against the Company, alleging violations of federal and state antitrust laws and California’s unfair competition law based on the Company’s operation of its App Store. The California District Court found that certain provisions of the Company’s App Store Review Guidelines violate California’s unfair competition law and issued an injunction preventing the Company from prohibiting developers from including external links in their apps that direct customers to purchasing mechanisms other than Apple in-app purchasing. The Company filed a compliance plan on January 16, 2024, and a motion to narrow or vacate the injunction on September 30, 2024.

4. **Other Legal Proceedings**: The Company is subject to various other legal proceedings and claims that have not been fully resolved and that have arisen in the ordinary course of business. The Company settled certain matters during the fourth quarter of 2024 that did not have a material impact on its financial condition or operating results.

The outcomes of these legal matters are inherently uncertain and could materially affect the Company's financial condition and operating results.

================================================================================
TEST COMPLETE!
================================================================================

Enter your question: What products does apple sell?

================================================================================
Question: What products does apple sell?
================================================================================
Retrieving relevant context...
Found 5 relevant chunks
  Chunk 1: Section=financials, Score=0.681
  Chunk 2: Section=financials, Score=0.681
  Chunk 3: Section=financials, Score=0.677
  Chunk 4: Section=financials, Score=0.674
  Chunk 5: Section=financials, Score=0.657

Generating answer...
ANSWER:
Based on the provided context, Apple Inc. sells a variety of products, including:

1. **Smartphones** - This includes the iPhone, which is a significant part of Apple's product lineup.
2. **Personal Computers** - This category encompasses products like the MacBook and iMac.
3. **Tablets** - Apple offers the iPad as part of its tablet offerings.
4. **Wearables** - This includes devices such as the Apple Watch and AirPods.
5. **Accessories** - Various accessories that complement their main products.
6. **Services** - Apple also provides a range of services, which may include digital content, software applications, and cloud services.

The company is focused on expanding its market opportunities related to these categories, as indicated in the context.

================================================================================
TEST COMPLETE!
================================================================================
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
