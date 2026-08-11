<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Movie%20Rating%20Predictor&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Will%20it%20be%20High-Rated%20on%20IMDb%3F&descAlignY=55&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![XGBoost](https://img.shields.io/badge/ROC--AUC-0.847-success?style=for-the-badge)](#-results)

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=18&pause=1000&color=2C5364&center=true&vCenter=true&width=650&lines=15%2C509+films+from+IMDb+Movies+India;5+pre-release+inputs%2C+78.3%25+accuracy;Tuned+XGBoost+beats+a+stacking+ensemble;3D+Three.js+predictor+%2B+results+dashboard)](https://git.io/typing-svg)

</div>

---

## 📖 Overview

Predicts whether a movie will be **High-Rated** (IMDb rating ≥ 6.5) using only five inputs available *before* release: primary **Genre**, **Director**, **Lead Actor**, **Release Year**, and **expected Vote count**. Trained on the **IMDb Movies India** dataset (15,509 films) and deployed behind a Flask app with a full **3D Three.js** predictor UI.

---

## 🏗️ Architecture

<div align="center">
<img src="architecture-svgs/03-movie-rating-architecture.svg" alt="Movie rating prediction animated architecture diagram" width="100%"/>
</div>

<details>
<summary>Mermaid source (fallback / editable)</summary>

```mermaid
flowchart LR
    A[("IMDb Movies India.csv<br/>15,509 films")] --> B["clean.py<br/>parse Year/Votes/Duration"]
    B --> C["features.py<br/>OOF target encoding + one-hot genre"]
    C --> D["train.py<br/>RandomizedSearchCV"]
    D --> E{Model Bakeoff}
    E --> F[Logistic Regression]
    E --> G[Random Forest]
    E --> H[Stacking Ensemble]
    E --> I[["XGBoost (tuned)<br/>— deployed"]]
    I --> J["models/ artifacts<br/>encoder + maps + metrics"]
    J --> K["Flask app.py<br/>/predict + /dashboard"]
    K --> L["3D Three.js predictor UI"]
    K --> M["Results Dashboard"]

    style A fill:#0f2027,color:#fff
    style I fill:#0f6b3a,color:#fff
    style L fill:#2c5364,color:#fff
    style M fill:#2c5364,color:#fff
```

</details>

---

## 📁 Project Structure

```text
├── src/
│   ├── clean.py       # parse + clean raw CSV, build High_Rated target
│   ├── features.py    # OOF smoothed target encoder, one-hot genre, matrix builder
│   └── train.py       # tune + train/compare models, save artifacts + dashboard plots
├── app/
│   ├── app.py         # Flask app: /predict and /dashboard
│   ├── templates/     # index.html (3D form), dashboard.html (results)
│   └── static/        # style.css, generated charts, Three.js scene
├── models/            # trained model, encoder, maps, metrics (generated)
├── run_app.py         # launch the web app
└── requirements.txt
```

---

## 🔬 Methodology

1. **Cleaning** — parse `Year` from `(2019)`, `Votes` from `"1,086"`, `Duration` from `"142 min"`; coerce `Rating` to numeric; drop unrated rows; build `High_Rated = Rating >= 6.5`.
2. **Feature engineering** — out-of-fold, smoothed target encoding for `Director`, `Lead Actor`, `Actor 2`, `Actor 3` (leakage-free); one-hot encode primary `Genre`; log-scale `Votes`; retain `Year`, `Duration`, genre count.
3. **Modeling** — stratified 80/20 split; `RandomizedSearchCV` (5-fold) tuning for XGBoost and Random Forest; a stacking ensemble compared against the tuned single model. Model selection happens on **out-of-fold predictions only** — the test split is touched exactly once.
4. **Deployment** — Flask app with an interactive 3D scene and a results dashboard.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.754 | 0.721 | 0.536 | 0.615 | 0.796 |
| Random Forest (tuned) | 0.759 | 0.724 | 0.552 | 0.627 | 0.827 |
| Stacking Ensemble | 0.782 | 0.745 | 0.614 | 0.673 | 0.847 |
| **XGBoost (tuned) — deployed** | **0.783** | **0.751** | **0.609** | **0.673** | **0.847** |
| Baseline (majority class) | 0.634 | — | — | — | 0.500 |

The deployed XGBoost reaches **78.3% accuracy** — a **+14.9 point lift** over the 63.4% majority-class baseline — using only a handful of pre-release inputs. The stacking ensemble matched it on F1/ROC-AUC, but out-of-fold selection kept the simpler tuned XGBoost as the production model.

---

## ⚡ Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Train
Point at the IMDb Movies India CSV ([Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)):
```bash
python -m src.train "path\to\IMDb Movies India.csv"
```
Writes the best model, fitted target encoder, genre/feature column maps, `metrics.json`, and dashboard charts into `models/`.

### Run the web app
```bash
python run_app.py
```
Open **http://127.0.0.1:5000**

---

## 🖥️ Using the App

- **Predictor (3D)** — a full Three.js scene with a starfield and a glowing probability meter that fills and changes color with every prediction; the form card tilts in 3D as you move the mouse. Enter genre, director, lead actor, year, and expected votes (plus optional duration/extra actors) for a `High-Rated` / `Not High-Rated` verdict with confidence. Unknown directors/actors fall back to the dataset prior.
- **Results Dashboard** — model comparison table, confusion matrix, ROC curves, feature importance, metric bars, and 50 sample test predictions.

> Three.js is bundled locally in `app/static/js/three.min.js`, so the 3D UI works fully offline.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB6423?style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Three.js](https://img.shields.io/badge/-Three.js-000000?style=flat-square&logo=three.js&logoColor=white)

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,100:0f2027&height=100&section=footer" width="100%"/>
</div>
