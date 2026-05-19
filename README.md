<div align="center">

<h1>🎬 CINEIQ</h1>
<h3>Explainable Movie Recommendation Engine</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p><em>A production-grade, end-to-end ML pipeline that recommends movies — and tells you <strong>exactly why</strong>.</em></p>

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
  - [Training the Models](#training-the-models)
  - [Running the Stack](#running-the-stack)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Dashboard Preview](#-dashboard-preview)
- [MLOps & Experiment Tracking](#-mlops--experiment-tracking)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

Most recommendation systems are **black boxes** — they suggest content but never explain *why*. CINEIQ changes that.

CINEIQ is a hybrid recommendation engine that combines **Collaborative Filtering**, **Content-Based Filtering**, and **Sentiment-Aware Re-Ranking** into a single, explainable pipeline. Every recommendation comes with a human-readable reason: *"Because you loved Inception's mind-bending plot and Christopher Nolan's direction..."*

Built for ML engineers and product teams who want a reference implementation of a real-world recommendation system with a full MLOps lifecycle.

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🤝 **Hybrid Recommendations** | Blends collaborative + content-based signals for best-in-class accuracy |
| 💬 **Explainability Layer** | Rule-based NLP generates plain-English explanations for every recommendation |
| 🎭 **Sentiment Re-Ranking** | VADER NLP adjusts scores using real audience review sentiment |
| 📊 **Taste Profile Dashboard** | Radar and bar charts visualize your genre preferences and viewing history |
| ⚡ **High-Performance API** | FastAPI backend with async endpoints serves predictions in milliseconds |
| 🧪 **MLOps Tracking** | MLflow logs all training runs, hyperparameters, and evaluation metrics |
| 🎯 **Cold-Start Handling** | Content-based fallback activates automatically for new users with no history |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CINEIQ Pipeline                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  MovieLens   │    │        TMDB 45K              │   │
│  │  25M Ratings │    │  (Metadata, Credits, Keys)   │   │
│  └──────┬───────┘    └───────────────┬──────────────┘   │
│         │                           │                   │
│         ▼                           ▼                   │
│  ┌─────────────┐           ┌────────────────┐           │
│  │Collaborative│           │ Content-Based  │           │
│  │  Filtering  │           │   Filtering    │           │
│  │(SVD Matrix) │           │(TF-IDF + CosSim│           │
│  └──────┬──────┘           └───────┬────────┘           │
│         │                         │                     │
│         └──────────┬──────────────┘                     │
│                    ▼                                     │
│          ┌──────────────────┐                           │
│          │  Hybrid Merger   │                           │
│          │  & Score Blender │                           │
│          └────────┬─────────┘                           │
│                   │                                     │
│                   ▼                                     │
│        ┌──────────────────────┐                         │
│        │  Sentiment Re-Ranker │◄── Audience Reviews     │
│        │    (NLTK VADER)      │                         │
│        └──────────┬───────────┘                         │
│                   │                                     │
│                   ▼                                     │
│        ┌──────────────────────┐                         │
│        │  Explainability Layer│                         │
│        │  (Rule-Based NLP)    │                         │
│        └──────────┬───────────┘                         │
│                   │                                     │
│         ┌─────────┴──────────┐                          │
│         ▼                    ▼                          │
│   ┌───────────┐      ┌───────────────┐                  │
│   │  FastAPI  │      │   Streamlit   │                  │
│   │  Backend  │      │   Dashboard   │                  │
│   └───────────┘      └───────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

### Machine Learning
- **[scikit-surprise](https://surpriselib.com/)** — SVD Matrix Factorization for Collaborative Filtering
- **[scikit-learn](https://scikit-learn.org/)** — TF-IDF vectorization and cosine similarity
- **[NLTK VADER](https://www.nltk.org/)** — Sentiment analysis for review-based re-ranking

### MLOps
- **[MLflow](https://mlflow.org/)** — Experiment tracking, parameter logging, and model registry

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** — Async REST API for model serving and explainability
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server

### Frontend
- **[Streamlit](https://streamlit.io/)** — Interactive user dashboard
- **[Plotly](https://plotly.com/python/)** — Radar and bar chart visualizations for taste profiles

### Data
- **[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)** — 25 million user ratings
- **[TMDB 45K Movies](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)** — Movie metadata, cast, crew, and keywords

---

## 📁 Project Structure

```
cineiq/
│
├── api/                        # FastAPI backend
│   ├── main.py                 # API entrypoint, routes
│   ├── recommender.py          # Prediction & hybrid merging logic
│   └── explainer.py            # Explainability string generation
│
├── dashboard/                  # Streamlit frontend
│   └── app.py                  # User dashboard & taste profile charts
│
├── data/
│   ├── raw/                    # Raw datasets (gitignored)
│   │   ├── ratings.csv
│   │   ├── links.csv
│   │   ├── movies_metadata.csv
│   │   ├── credits.csv
│   │   └── keywords.csv
│   └── processed/              # Cleaned & feature-engineered data
│
├── models/                     # Serialized trained models (gitignored)
│   ├── svd_model.pkl
│   └── tfidf_matrix.pkl
│
├── notebooks/                  # EDA and prototyping notebooks
│   ├── 01_eda.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   └── 03_content_based.ipynb
│
├── src/                        # Core ML pipeline modules
│   ├── collaborative.py        # SVD training & prediction
│   ├── content_based.py        # TF-IDF & cosine similarity
│   ├── sentiment.py            # VADER sentiment re-ranker
│   └── preprocess.py           # Data cleaning & feature engineering
│
├── mlruns/                     # MLflow tracking directory (gitignored)
├── run_pipeline.py             # End-to-end training orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip / virtualenv
- ~5 GB free disk space for datasets
- Git

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/cineiq.git
cd cineiq
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** NLTK requires a one-time download of the VADER lexicon. The pipeline handles this automatically on first run, or you can do it manually:
> ```python
> import nltk
> nltk.download('vader_lexicon')
> ```

---

### Data Setup

CINEIQ requires two public datasets. Download them and place files as shown below.

**Dataset 1 — MovieLens 25M**

Download from [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/)

Place in `data/raw/`:
```
data/raw/ratings.csv
data/raw/links.csv
```

**Dataset 2 — TMDB 45K Movies (Kaggle)**

Download from [Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Place in `data/raw/`:
```
data/raw/movies_metadata.csv
data/raw/credits.csv
data/raw/keywords.csv
```

Your `data/raw/` directory should now contain all 5 files before proceeding.

---

### Training the Models

Run the full end-to-end training pipeline. This will preprocess data, train the SVD and TF-IDF models, run sentiment analysis, and log everything to MLflow.

```bash
python run_pipeline.py
```

Expected output:
```
[1/4] Preprocessing data...          ✓
[2/4] Training SVD (Collaborative)... ✓  RMSE: 0.8731
[3/4] Building TF-IDF matrix...       ✓  Vocabulary: 12,847 terms
[4/4] Logging run to MLflow...        ✓  Run ID: a3f2b1...
Pipeline complete. Models saved to /models
```

> ⚠️ Training on the full 25M dataset can take **20–40 minutes** depending on your hardware. Use a subset for faster iteration during development (see notebooks/).

---

### Running the Stack

Open **two terminal windows** (both with the virtual environment activated):

**Terminal 1 — Start the FastAPI backend**

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`
Interactive docs available at `http://localhost:8000/docs`

**Terminal 2 — Start the Streamlit dashboard**

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## ⚙️ How It Works

### Step 1 — Collaborative Filtering (SVD)

Using **Singular Value Decomposition** via `scikit-surprise`, the engine decomposes the user-item rating matrix to learn latent factors. For a given user, it predicts ratings for unseen movies and returns top-N candidates.

```
R ≈ U × Σ × Vᵀ
```

Where `U` = user latent factors, `V` = item latent factors, `Σ` = singular values.

### Step 2 — Content-Based Filtering (TF-IDF)

A **"soup" feature** is constructed by concatenating a movie's genres, top cast members, director, and keywords into a single string. TF-IDF vectorization converts these into weighted term vectors, and **cosine similarity** finds the most similar movies to a user's favorites.

### Step 3 — Hybrid Merging

Scores from both models are **weighted and combined**. The default blend is:
- 60% Collaborative Filtering
- 40% Content-Based Filtering

Weights are configurable via `api/config.py`.

### Step 4 — Sentiment Re-Ranking

NLTK's **VADER** (Valence Aware Dictionary and sEntiment Reasoner) analyzes audience review text for each candidate movie. Movies with overwhelmingly positive sentiment are boosted; critically panned films are penalized.

### Step 5 — Explainability

The **Explainability Layer** inspects the highest-contributing signals for each recommendation and generates a human-readable string:

> *"Recommended because you gave The Dark Knight 5 stars — Interstellar shares the same director (Christopher Nolan), sci-fi genre, and is loved by viewers with your taste profile (avg. audience sentiment: 87% positive)."*

---

## 📡 API Reference

Once the backend is running, full interactive docs are at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/recommendations/{user_id}` | Get top-N recommendations for a user |
| `GET` | `/similar/{movie_id}` | Find movies similar to a given title |
| `GET` | `/user/{user_id}/profile` | Fetch a user's genre taste profile |
| `GET` | `/health` | API health check |

**Example Request**

```bash
curl "http://localhost:8000/recommendations/42?n=5"
```

**Example Response**

```json
{
  "user_id": 42,
  "recommendations": [
    {
      "movie_id": 157336,
      "title": "Interstellar",
      "predicted_rating": 4.7,
      "explanation": "Matches your love of Christopher Nolan's direction and sci-fi themes.",
      "sentiment_score": 0.89
    }
  ]
}
```

---

## 📊 Dashboard Preview

The Streamlit dashboard provides:

- **Taste Profile Radar Chart** — Visualizes genre affinity across 15+ categories
- **Top Rated Movies** — Your personal highest-rated films
- **Live Recommendations** — Fetches and displays real-time predictions with explanations
- **Sentiment Gauge** — Shows audience sentiment score for each recommended title

---

## 🧪 MLOps & Experiment Tracking

All training runs are tracked with **MLflow**. To launch the MLflow UI:

```bash
mlflow ui
```

Then visit `http://localhost:5000` to explore:

- Model hyperparameters (SVD factors, epochs, learning rate)
- Evaluation metrics (RMSE, MAE, Precision@K)
- Artifacts (serialized models, vectorizers)
- Run comparison across experiments

**Key tracked parameters:**

| Parameter | Default |
|-----------|---------|
| `svd_n_factors` | 100 |
| `svd_n_epochs` | 20 |
| `svd_lr_all` | 0.005 |
| `hybrid_collab_weight` | 0.6 |
| `sentiment_boost_factor` | 0.15 |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/add-deep-learning-model`
3. **Commit your changes**: `git commit -m "feat: add neural collaborative filtering"`
4. **Push to the branch**: `git push origin feature/add-deep-learning-model`
5. **Open a Pull Request**

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and ensure all tests pass before submitting a PR.

**Ideas for contributions:**
- [ ] Neural Collaborative Filtering (NCF) model
- [ ] A/B testing framework for hybrid weights
- [ ] Docker Compose setup for one-command deployment
- [ ] User authentication in the Streamlit dashboard
- [ ] Caching layer (Redis) for API responses

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [AyuuRaj](https://github.com/AyuuRaj)

⭐ **Star this repo** if you found it useful!

</div>
