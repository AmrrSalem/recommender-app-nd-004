Basic Recommender System
==========================

Overview
--------
This project is a basic recommender system implemented with FastAPI and FastHTML.
It demonstrates multiple recommendation approaches required by the rubric:

1. Rank-based (popularity) recommendations
2. User–User Collaborative Filtering (CF)
3. Content-based (TF-IDF + clustering)
4. Matrix Factorization (SVD latent factors)
5. Offline Evaluation (Precision@k, Recall@k, F1@k, MAP@k, NDCG@k, Item Coverage)

Input Data
----------
The app requires only one file:
    data/interactions.csv

This CSV must contain at least:
    user_id (or "email" will be converted to user_id)
    article_id
    title (optional, a fallback title is generated if missing)

Any additional files are generated internally (e.g., articles are derived in memory).

Installation
------------
1. Create and activate a virtual environment (recommended).

   Windows (PowerShell):
       python -m venv .venv
       .venv\Scripts\activate

   Linux / Mac:
       python3 -m venv .venv
       source .venv/bin/activate

2. Upgrade pip and tools:
       python -m pip install --upgrade pip setuptools wheel

3. Install dependencies:
       pip install -r requirements.txt

   Notes:
   - python-fasthtml==0.4.3 is optional. If installation fails, you may comment it out
     in requirements.txt. The app will fall back to a built-in HTML renderer.

Running the App
---------------
1. Ensure your data file exists:
       data/interactions.csv

2. Start the application:
       python -m uvicorn app:app --reload

3. Open the local web interface in your browser:
       http://127.0.0.1:8000

Features
--------
- User Recommendations:
    Enter a user_id and number of items (n) to see recommendations.
    CF is used when possible, otherwise popularity fallback.

- Article Recommendations:
    Enter an article_id and number of items (n).
    Shows both Content-based (TF-IDF similarity) and SVD-based recommendations.

- Evaluation:
    Run offline evaluation (leave-one-out, sampled users).
    Reports Precision@k, Recall@k, F1@k, MAP@k, NDCG@k, and Item Coverage
    for all four models (Popularity, CF, Content, SVD).

Project Requirements
--------------------
- Rank-based, CF, Content-based, and SVD recommenders are implemented.
- Cold-start fallback to popularity is included.
- Evaluation metrics are provided via /eval endpoint and web UI.
- Web interface created with FastAPI + FastHTML (with fallback if not installed).

Optional Extensions
-------------------
- Package the recommender as a pip-installable library
- Extend content-based recommendations with advanced NLP methods

License
-------
This project is for learning.
