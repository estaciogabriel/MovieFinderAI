## DOC - MOVIEKNOWLEDGEBASE

This repository contains the code for a Movie Knowledge Base chatbot application.  

# Movies Knowledge Base

Semantic search system for movies using embeddings and Chroma Cloud. Enter details like context, characters, or actors, and it will find the most relevant matches.

```bash
   Leonardo Dicaprio in a movie that he build plans
```

```bash
   The Amazing Spider-Man (2012)
    Genres: Action, Adventure, Fantasy
    Director: Marc Webb
    Cast: Andrew Garfield, Emma Stone, Rhys Ifans, Denis Leary, Campbell Scott
    Rating: 6.5/10 (7K votes)
    Runtime: 136 min
    Peter Parker is an outcast high schooler abandoned by his parents as a boy, leaving him to be raised by his Uncle Ben and Aunt May. Like most teenagers, Peter is trying to figure out who he is and how he got to be the person he is today. As Peter discovers a mysterious briefcase that belonged to his father, he begins a quest to understand his parents' disappearance – leading him directly to Oscorp and the lab of Dr. Curt Connors, his father's former partner. As Spider-Man is set on a collision course with Connors' alter ego, The Lizard, Peter will make life-altering choices to use his powers and shape his destiny to become a hero.
    Distance: 0.8308
```





## Execution 

**Requiriments** IMPORTANT

 "python3 -m venv .venv"

 "uv pip install --python .venv/bin/python -r pyproject.toml"

 "cp .env.example .env"  (fill in CHROMA_API_KEY)

**ChatBot**

 "make run"              (src/interfaces/pages/gradio_app.py)
 
**Dashboard**
 
 "make run-dashboard"    (src/interfaces/pages/dashboard.py)

 **Pipiline**

 ".venv/bin/python -m scripts.pipeline"

 

## Introduction

A system that allows searching for movies using natural language. Type "action movie with a chase scene" and find similar movies by meaning, not just exact words.

## How It Works

1. 43,970 movies from Kaggle transformed into embeddings (384-dimensional vectors)
2. Stored in Chroma Cloud (vector database)
3. Semantic similarity search using sentence-transformers

## Technologies

- Python 3.12
- Sentence-Transformers (all-MiniLM-L6-v2)
- Chroma Cloud
- Gradio + Streamlit
- Scikit-learn
