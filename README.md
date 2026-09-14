we are building an SMS Spam Classifier from scratch

This python code uses Multinomial Naive Bayes with add-1 (Laplace) smoothing to track spam and ham(not spam) SMS

Rules: unknown words (never seen in training) are dropped; add-1 smoothing keeps every  P(w∣c)>0  so a single missing word can't zero out a class.

Heads-up: real data is imbalanced — about 87% of messages are ham. So unlike a toy 50/50 set, the prior genuinely matters here.
