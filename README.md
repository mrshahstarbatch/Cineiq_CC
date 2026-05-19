# CINEIQ

## Intelligent Movie Recommendation & Analytics Platform

CINEIQ is a movie recommendation platform focused on delivering personalized film suggestions using user behavior, movie metadata, and preference analysis.

The system combines multiple recommendation approaches to improve recommendation quality while also providing interactive visual insights through a dashboard interface.

---

## Features

- Personalized movie recommendations
- Hybrid recommendation system
- Interactive analytics dashboard
- Genre and preference visualization
- Similar movie discovery
- Sentiment-aware ranking
- FastAPI backend integration
- Streamlit-based dashboard

---

## Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-Learn
- TF-IDF Vectorization
- Collaborative Filtering
- NLP Sentiment Processing

### Frontend
- Streamlit
- Plotly

---

## Project Structure

```text
CINEIQ/
│
├── api/
├── dashboard/
├── ml_engine/
├── data/
├── models/
├── notebooks/
├── config.py
├── run_pipeline.py
└── requirements.txt
```

---

## Recommendation Pipeline

The recommendation workflow includes:

1. User preference analysis  
2. Collaborative filtering  
3. Content-based similarity matching  
4. Sentiment-based re-ranking  
5. Recommendation explanation generation  

---

## Dashboard

The dashboard provides:

- User preference insights
- Genre distribution analysis
- Recommendation visualizations
- Interactive movie exploration

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/CINEIQ.git

cd CINEIQ

pip install -r requirements.txt
```

---

## Run Backend

```bash
uvicorn api.main:app --reload
```

---

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

- Real-time recommendation updates
- User authentication
- Advanced recommendation tuning
- Docker deployment support
- Cloud model hosting

---

## License

This project is intended for educational and development purposes.
