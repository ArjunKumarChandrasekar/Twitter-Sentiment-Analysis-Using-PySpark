# 🐦 Twitter Sentiment Analysis using PySpark

A machine learning system that reads tweets and figures out whether the person writing them was in a good mood, a bad mood, or somewhere in between. Built with PySpark to handle Twitter-scale data without breaking a sweat.

---

## 😤 The Problem

Every single day, hundreds of millions of tweets go out into the world. People venting about a flight delay. Fans cheering after a game. Customers complaining about a product. Voters sharing opinions on a policy.

All of that is raw, unfiltered public opinion — and it's incredibly valuable. But nobody can read millions of tweets manually.

The question this project answers: **Can we build a system that automatically reads tweets and understands whether they're positive, negative, or neutral?**

And more importantly — can we do it at a scale that actually matches how fast Twitter moves?

---

## 💡 The Solution

Use **PySpark** (Apache Spark's Python interface) combined with a **Logistic Regression** classifier to process tweets in bulk and label each one with a sentiment: positive, negative, or neutral.

PySpark is designed for big data — it splits work across multiple processors and handles huge datasets efficiently. Pair that with a solid NLP pipeline and you have a system that can chew through Twitter data at scale.

---

## ⚙️ How It Works

**1. Load the data**
The project comes with `twtr_dataset.csv` — a labeled dataset of real tweets, each tagged with its sentiment. This is what the model learns from.

**2. Clean and prepare the tweets**
Raw tweets are messy. They have hashtags, @mentions, links, slang, and weird capitalization. Before any analysis, the text gets cleaned up — stripped of noise so the model can focus on the actual words and meaning.

**3. Convert text into numbers**
Machines don't understand words — they understand numbers. The text is transformed using NLP techniques (like tokenization and TF-IDF) that turn each tweet into a set of numerical features the model can work with.

**4. Train a Logistic Regression model**
The cleaned, vectorized tweets are fed into a Logistic Regression classifier inside PySpark's MLlib. The model learns to associate certain word patterns with positive or negative sentiment.

**5. Predict and evaluate**
Once trained, the model predicts the sentiment of new tweets and reports how accurate it is — giving a clear picture of how well it's performing.

Everything runs inside `pyspark_logistic_reg_twitter.ipynb`.

---

## 🎯 Use Cases

Once you can classify tweets at scale, a lot of things become possible:

- **Brand monitoring** — Track what people are saying about your company or product in real time. Is the sentiment going up or down after a product launch?
- **Crisis detection** — Spot a sudden spike in negative tweets about your service before it becomes a PR disaster
- **Political & social research** — Measure public opinion around elections, policies, or social movements
- **Stock market signals** — Sentiment around a company's ticker often moves before the stock does
- **Customer feedback at scale** — Instead of reading support tickets one by one, classify thousands of them instantly
- **Event monitoring** — Understand how audiences feel about a live event, movie release, or sports match as it unfolds

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Big Data Framework | Apache Spark (PySpark) |
| Machine Learning | PySpark MLlib — Logistic Regression |
| NLP Processing | Tokenizer, StopWordsRemover, TF-IDF (HashingTF + IDF) |
| Dataset | `twtr_dataset.csv` (labeled tweets) |
| Environment | Jupyter Notebook |
| Language | Python |

---

## 🚀 Getting Started

**Prerequisites**

```bash
pip install pyspark
pip install jupyter
```

You'll also need Java installed (required for Apache Spark to run).

**Run the notebook**

```bash
jupyter notebook pyspark_logistic_reg_twitter.ipynb
```

Open it and run the cells from top to bottom. The notebook walks through every step — loading data, preprocessing, training, and evaluating the model.

---

## 📁 Project Structure

```
├── pyspark_logistic_reg_twitter.ipynb   # Full ML pipeline in PySpark
├── twtr_dataset.csv                     # Labeled Twitter dataset
└── README.md
```

---

## 🔍 Why PySpark?

You might wonder — why not just use regular pandas or scikit-learn?

The answer is scale. A typical Twitter dataset can have millions of rows. Regular Python tools start choking at that size. PySpark distributes the work across cores (or even across machines in a cluster), so the same code that works on your laptop could also run on a cloud cluster processing billions of tweets. It's the right tool for the job.

---

## 📊 How Sentiment is Classified

| Label | Meaning |
|---|---|
| Positive | Tweet expresses happiness, satisfaction, or enthusiasm |
| Negative | Tweet expresses frustration, anger, or disappointment |
| Neutral | Tweet is factual, informational, or doesn't lean either way |

---

## 🌱 What Could Come Next

- Adding a real-time Twitter stream using the Twitter API to classify live tweets as they come in
- Trying more powerful models like Naive Bayes or even a fine-tuned BERT for higher accuracy
- Building a simple dashboard that visualizes sentiment trends over time
- Expanding beyond binary/ternary sentiment into emotion detection (joy, anger, fear, surprise)
