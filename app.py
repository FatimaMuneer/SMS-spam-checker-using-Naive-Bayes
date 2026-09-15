import streamlit as st
import math
import re
import os
import urllib.request
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Classifier (from Scratch)",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------------------
# 1. Custom Model Classes
# -------------------------------------------------------------
class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        self.vocab = set()
        self.doc_count = {c: 0 for c in self.classes}
        self.word_counts = {c: {} for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        n_docs = len(docs)

        for text, c in zip(docs, labels):
            tokens = self._tokenize(text)
            self.doc_count[c] += 1
            for w in tokens:
                self.vocab.add(w)
                self.word_counts[c][w] = self.word_counts[c].get(w, 0) + 1
                self.total_words[c] += 1

        self.vocab_size = len(self.vocab)
        self.priors = {c: self.doc_count[c] / n_docs for c in self.classes}
        return self

    def _likelihood(self, word, c):
        count = self.word_counts[c].get(word, 0)
        return (count + self.alpha) / (self.total_words[c] + self.alpha * self.vocab_size)

    def predict_one_prob(self, text):
        tokens = self._tokenize(text)
        score = self.priors["spam"]
        for w in tokens:
            if w in self.vocab:
                score *= self._likelihood(w, "spam")
        return score

class LogNaiveBayes(MultinomialNaiveBayes):
    def predict_one(self, text):
        tokens = self._tokenize(text)
        scores = {}
        for c in self.classes:
            score = math.log(self.priors[c])
            for w in tokens:
                if w in self.vocab:
                    score += math.log(self._likelihood(w, c))
            scores[c] = score
        best_class = max(scores, key=scores.get)
        return best_class, scores

# -------------------------------------------------------------
# 2. Data Loading and Caching
# -------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    path = "sms_spam.tsv"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    docs, labels = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        label, text = line.split("\t", 1)
        labels.append(label)
        docs.append(text)

    # Train from-scratch log model
    model = LogNaiveBayes(alpha=1).fit(docs, labels)

    # Calculate top spam words using odds ratio
    ratios = {w: model._likelihood(w, "spam") / model._likelihood(w, "ham") for w in model.vocab}
    top_words = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:15]

    return model, len(docs), labels.count("spam"), labels.count("ham"), top_words

model, total_docs, spam_count, ham_count, top_words = load_and_train_model()

# -------------------------------------------------------------
# 3. Streamlit Interface
# -------------------------------------------------------------
st.title("🛡️ SMS Spam Classifier — From Scratch")
st.markdown(
    "A custom **Multinomial Naive Bayes** implementation built entirely with Python primitives "
    "(without external ML libraries for inference), featuring **Laplace Smoothing** and **Log-Space Transformation**."
)

# Sidebar Metrics
st.sidebar.header("Dataset & Model Overview")
st.sidebar.metric("Total Messages", total_docs)
st.sidebar.metric("Class Prior (Ham)", f"{model.priors['ham'] * 100:.1f}%")
st.sidebar.metric("Class Prior (Spam)", f"{model.priors['spam'] * 100:.1f}%")
st.sidebar.metric("Vocabulary Size", f"{model.vocab_size:,} words")
st.sidebar.metric("Smoothing Factor (α)", model.alpha)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Live Prediction", "📊 Indicative Vocabulary", "⚠️ Underflow Simulator"])

with tab1:
    st.subheader("Test a Custom Message")
    sample_choice = st.selectbox(
        "Choose a sample message or type your own below:",
        [
            "Custom text...",
            "congratulations you won a free prize call now to claim",
            "hey are we still on for dinner tonight",
            "URGENT your account has won 1000 cash reply now"
        ]
    )

    user_input = st.text_area(
        "Enter SMS text:",
        value="" if sample_choice == "Custom text..." else sample_choice,
        placeholder="Type or paste an SMS message here..."
    )

    if st.button("Classify Message", type="primary"):
        if user_input.strip():
            prediction, log_scores = model.predict_one(user_input)
            tokens = model._tokenize(user_input)
            recognized_tokens = [w for w in tokens if w in model.vocab]

            # Display prediction banner
            if prediction == "spam":
                st.error(f"### Result: **SPAM** 🚨")
            else:
                st.success(f"### Result: **HAM (Legitimate)** ✅")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model Log Scores:**")
                st.write(f"- $\\log P(\\text{{spam}} \\mid \\text{{msg}})$: `{log_scores['spam']:.4f}`")
                st.write(f"- $\\log P(\\text{{ham}} \\mid \\text{{msg}})$: `{log_scores['ham']:.4f}`")
            with col2:
                st.markdown("**Token Inspection:**")
                st.write(f"- Extracted words: `{tokens}`")
                st.write(f"- Recognized in vocabulary: `{recognized_tokens}`")
        else:
            st.warning("Please enter a message to analyze.")

with tab2:
    st.subheader("Strongest Spam-Indicative Words")
    st.markdown(
        "Calculated via likelihood ratio: "
        "$\\text{Ratio}(w) = \\frac{P(w \\mid \\text{spam})}{P(w \\mid \\text{ham})}$"
    )
    df_words = pd.DataFrame(top_words, columns=["Word", "Spam Likelihood Ratio"])
    df_words["Spam Likelihood Ratio"] = df_words["Spam Likelihood Ratio"].round(2)
    st.dataframe(df_words, use_container_width=True)

with tab3:
    st.subheader("Why the Log Trick Matters (Arithmetic Underflow)")
    st.markdown(
        "When processing long text sequences, multiplying probabilities ($0 < P < 1$) repeatedly collapses the "
        "final float toward zero until Python sets it to `0.0`. Log-space converts multiplication to addition."
    )

    repeat_count = st.slider("Number of concatenated spam messages:", min_value=1, max_value=12, value=6)
    simulated_spam = "claim prize free urgent cash won text call " * repeat_count
    
    raw_prob = model.predict_one_prob(simulated_spam)
    _, log_scores = model.predict_one(simulated_spam)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Raw Probability Product", f"{raw_prob}")
        if raw_prob == 0.0:
            st.error("Underflow occurred! Raw product collapsed to 0.0.")
    with col_b:
        st.metric("Log-Space Score", f"{log_scores['spam']:.2f}")
        st.success("Safe numerical range maintained.")
