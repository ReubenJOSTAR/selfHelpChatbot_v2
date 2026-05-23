# WalkWithJesus ✝️

AI-powered Christian guidance assistant built using Retrieval Augmented Generation (RAG), semantic search, and emotionally contextual scripture retrieval.

---

# Overview

WalkWithJesus is an experimental AI system designed to provide scripture-grounded encouragement and emotionally relevant Bible verses using semantic retrieval techniques.

Unlike traditional keyword-based Bible search systems, this project focuses on:

- semantic emotional understanding
- conversational retrieval
- metadata-aware vector search
- grounded scripture retrieval
- emotionally contextual AI systems

The goal is to create an AI assistant that can retrieve meaningful scripture based on how people naturally express emotions and struggles.

Example:

```text
"I feel overwhelmed and anxious about my future."
```

can semantically retrieve verses such as:

```text
Philippians 4:6-7
Matthew 6:25-34
```

without requiring exact keyword matches.

---

# Current Architecture

```text
User Query
    ↓
OpenAI Embedding
    ↓
Vector Similarity Search
    ↓
Metadata Filtering
    ↓
Semantic Retrieval
    ↓
Retrieved Scripture Context
```

---

# Technologies Used

## Backend

- Python
- LangChain
- OpenAI Embeddings
- MongoDB Atlas Vector Search
- FAISS (planned for local-first development)

## Planned Stack

- FastAPI
- React / Next.js
- TailwindCSS

---

# Retrieval Philosophy

This project prioritizes:

```text
retrieval quality over framework complexity
```

The retrieval layer is designed around:

- conversational phrasing
- emotional context
- semantic representation
- human intent matching

instead of relying only on sparse keywords.

---

# Dataset Design

Each scripture entry includes:

```json
{
  "feeling": "Anxiety",
  "reference": "Philippians 4:6-7",
  "message": "...",
  "search_text": "...",
  "semantic_context": "..."
}
```

## Important Concepts

### `search_text`

Contains emotionally relevant semantic keywords.

### `semantic_context`

Contains conversational emotional phrasing representing real-life human situations.

This improves:

- semantic retrieval quality
- emotional relevance
- conversational matching
- embedding representation

---

# Features Implemented

✅ OpenAI embedding pipeline  
✅ Semantic vector retrieval  
✅ Metadata-aware retrieval  
✅ Similarity score evaluation  
✅ Emotionally contextual dataset structure  
✅ Retrieval debugging pipeline  
✅ Modular ingestion architecture  

---

# Current Development Focus

The current focus is:

- improving retrieval quality
- semantic representation
- grounding quality
- emotionally relevant retrieval

before scaling infrastructure or adding advanced orchestration.

---

# Future Roadmap

## Planned Features

- LLM response generation
- conversational memory
- FastAPI backend
- frontend chat interface
- user personalization
- hybrid retrieval pipelines
- local vector store optimization

---

# Why This Project Exists

This project was built to explore:

- embeddings
- semantic search
- retrieval engineering
- grounded AI systems
- emotionally contextual AI assistants
- Christian AI guidance systems

while learning modern AI engineering concepts deeply rather than relying entirely on tutorials.

---

# Development Philosophy

WalkWithJesus follows a:

```text
local-first, retrieval-first
```

development strategy.

The system aims to:

1. perfect retrieval quality first
2. improve semantic understanding
3. optimize contextual grounding
4. add generation later
5. scale infrastructure only after architecture stabilizes

---

# License

MIT License

© 2026 Reuben Joseph

---

# Disclaimer

WalkWithJesus is an experimental AI engineering project and should not be considered a replacement for:

- pastoral counseling
- professional mental health support
- spiritual leadership
- theological authority

The system is designed for educational and experimental purposes.
