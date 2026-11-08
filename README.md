# Low SES Text Classification

**My classifier comparison from SMU research on spotting low socioeconomic status (SES) narratives in Reddit text: nine models, from Naive Bayes to an LSTM and a CNN, on the same human-labeled set. This work fed into a CoNLL 2025 paper where I am second author.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Links:** [Paper on the ACL Anthology](https://aclanthology.org/2025.conll-1.23/) · [Results](#results) · [Notebooks](#project-structure) · [Arham Doshi](https://arhamdoshi.dev) · [LinkedIn](https://linkedin.com/in/arhamdoshi)

"Bridging the Socioeconomic Gap in Education: A Hybrid AI and Human Annotation Approach," CoNLL 2025.

## Results

Every number below is in the saved outputs of [`Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes_KNN_Decision_Tree.ipynb`](Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes_KNN_Decision_Tree.ipynb). The data is 128 human-labeled texts, half low SES and half not, with 20% (26 texts) held out for testing.

| Model | Accuracy | F1 |
|---|---|---|
| Multinomial Naive Bayes | 0.81 | **0.83** |
| Logistic regression | 0.81 | 0.76 |
| SVM | 0.81 | 0.76 |
| Random forest | 0.65 | 0.69 |
| Gradient boosting | 0.65 | 0.61 |
| Decision tree | 0.62 | |
| 1D CNN (Keras) | 0.58 | |
| KNN | 0.46 | |
| LSTM (Keras) | 0.38 | |

On this small set, simple bag-of-words models beat the neural ones, which had too little data to learn from. The full paper, on a much larger corpus, reached the same overall conclusion: classical models with good features beat fine-tuned BERT and few-shot LLMs on this task.

## Why it matters

Students from low-income backgrounds often describe their situation indirectly, and those signals get missed. Finding them in text at scale could help direct support to the people who need it, but only if the classifier is trustworthy. Comparing many models side by side on the same labels shows which approaches hold up on small, carefully labeled data, which is what most real annotation projects have.

## How it works

1. **Data.** Low SES and not-low-SES narratives drawn from Reddit and labeled by people, balanced to 64 of each.
2. **Features.** NLTK stopword removal and tokenization, then a bag-of-words `CountVectorizer`, or a Keras tokenizer with padded sequences for the neural models.
3. **Models.** Naive Bayes, logistic regression, SVM, KNN, decision tree, random forest, and gradient boosting in scikit-learn, plus an LSTM and a 1D CNN in Keras, with class weights. A BERT fine-tuning setup is in the code too.
4. **Evaluation.** The same 80/20 split for every model, scored with accuracy, precision, recall, and F1.

| Choice | Why |
|---|---|
| One shared split for every model | The comparison is fair only if every model sees the same test texts |
| F1 next to accuracy | Precision and recall can trade off sharply on a small test set, and F1 shows that |
| Classical models alongside neural ones | With 128 examples, a model that needs less data can win, and here it did |

My notebooks cover the supervised classifier comparison. The paper's TF-IDF, PCA, and K-means clustering of about 799,000 Reddit texts and its few-shot LLaMA and OPT experiments were part of the wider project and are not in this repo.

## Tech stack

Python, scikit-learn, TensorFlow and Keras, PyTorch with Hugging Face Transformers (BERT), NLTK, pandas, Google Colab.

## Getting started

The notebooks were written in Google Colab and read the labeled text files from a Google Drive folder (`My Drive/SMU Work`).

1. Open a notebook in Colab: [full comparison](https://colab.research.google.com/github/hammy229/SMU-Low-SES-Extraction-Work/blob/main/Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes_KNN_Decision_Tree.ipynb), [earlier version](https://colab.research.google.com/github/hammy229/SMU-Low-SES-Extraction-Work/blob/main/Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes.ipynb), or [first baseline](https://colab.research.google.com/github/hammy229/SMU-Low-SES-Extraction-Work/blob/main/Arham_Low_ses_simple_classifier.ipynb). Or download one and run it in Jupyter.
2. Point the data path in the first cells at your own copy of the labeled text files. The research data is not public.
3. Run all cells.

## Project structure

```
Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes_KNN_Decision_Tree.ipynb   the full nine-model comparison
Doshi_Low_SES_LR_LSTM_CNN_SVM_Naive_Bayes.ipynb                     an earlier version with fewer models
Arham_Low_ses_simple_classifier.ipynb                               the first baseline classifiers
```

## Testing

This is a research notebook repo, so there are no automated tests. The notebooks keep their saved outputs, so every result above can be checked without rerunning anything.

## Limitations and next steps

- The test set is 26 texts, so one text changes accuracy by about 4 points. The ranking between close models is not reliable.
- The labeled data is not public, so the notebooks cannot be rerun as is.
- The neural models were underfed: 128 examples is far too few for an LSTM to learn from.
- No cross-validation. Repeating the split many times would give a much steadier comparison.

## License

MIT, see [LICENSE](LICENSE). Built by [Arham Doshi](https://arhamdoshi.dev).
