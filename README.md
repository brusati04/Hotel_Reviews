# Hotel Review Classification with a Bidirectional LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JC3UCnJe6MmMH7Hib4JfCBC6WnFQhMNV?usp=sharing)

A deep learning model that classifies hotel reviews as **good** or **bad**. It combines the review text, encoded by a **Bidirectional LSTM**, with the numeric review score.

Built for the *Machine Learning, Artificial Neural Networks and Deep Learning* exam (BSc in Artificial Intelligence, University of Pavia), June 2025.

## Results (test set)

| Metric | Value |
|---|---|
| Accuracy | **0.990** |
| F1-score | **0.990** |
| Confusion matrix | `[[355, 0], [7, 327]]` |

> **Note:** the review score is also an input to the model, and it is strongly correlated with the good/bad label. That is a large part of why the scores are so high. A text-only version of the model is the natural next experiment.

## Data

`input_data.csv` has 13,772 hotel reviews (balanced: 6,888 bad / 6,884 good). Columns include hotel name and address, reviewer nationality, review date, `Review` (free text, up to 400 words), `Review_Score` (0–10) and `Review_Type` (target).

## Approach

1. **Preprocessing**: keep `Review`, `Review_Score` and `Review_Type`. Lowercase the text and strip punctuation, then tokenize, and pad each review to a `max_len` that covers 95% of reviews. Labels are encoded as 0/1.
2. **Model** (Keras functional API, two inputs):
   - text → `Embedding` → `Bidirectional(LSTM(64))`
   - concatenated with the review score
   - → `Dense(64)` → `Dense(32)` (L2, batch norm, dropout) → `sigmoid` output
3. **Training**: binary cross-entropy with Adam. A small random search over learning rate, dropout, activation, epochs and batch size picks the configuration with the best validation F1.
4. **Evaluation**: train/validation/test split, F1, accuracy, confusion matrix and training curves.

## Run it

Click the Colab badge above, or run it locally:

```bash
pip install -r requirements.txt
jupyter notebook hotel_reviews_bilstm.ipynb
```

## License

[MIT](LICENSE) © Lorenzo Brusati
