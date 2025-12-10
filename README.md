# Emotion Classification Assistant

A text-based AI tool that predicts fine-grained emotions from user comments. This project evaluates multiple machine learning approaches, including TF-IDF with Logistic Regression, TF-IDF with Linear SVM, and GloVe embeddings with a feedforward neural network, using the GoEmotions dataset.

## Project Overview

The Emotion Classification Assistant is designed to analyze short text comments and identify the dominant emotion expressed. Unlike basic sentiment analysis, this project predicts one of 27 emotion classes plus neutral, covering a wide spectrum of human emotional expression such as joy, sadness, anger, surprise, admiration, and curiosity.

Key goals:
- Compare traditional machine learning methods and neural network models for emotion classification.
- Identify the most effective approach based on accuracy, macro F1-score, and confusion matrices.
- Provide a simple demonstration interface for testing real-time predictions.

## Data

This project uses the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions), a collection of 58,009 English-language Reddit comments labeled with 27 emotions plus neutral. The dataset is preprocessed to:
- Merge all raw CSV files into a single dataset
- Filter out comments with low annotator agreement
- Clean and tokenize text
- Assign a dominant emotion label per comment

## Models

Three models were evaluated:

- **Model A: TF-IDF + Logistic Regression**  
  Converts text to TF-IDF vectors and uses logistic regression to predict emotions. Best overall performance with balanced predictions across frequent and rare emotions.  

- **Model B: TF-IDF + Linear SVM**  
  Uses TF-IDF vectors with a linear SVM. Performs similarly to Model A but slightly lower overall accuracy, with weaker performance on rare emotions.  

- **Model C: GloVe Embeddings + Neural Network**  
  Represents words using 100-dimensional GloVe embeddings, averages into comment vectors, and trains a small feedforward neural network. Moderate performance on common emotions, low performance on rare emotions.

## Results

- **Model A** achieved the best balance of accuracy (51%) and macro F1-score (0.45).  
- Model B had slightly lower performance (Accuracy 49%, Macro F1 0.41).  
- Model C struggled with rare emotions (Accuracy 32%, Macro F1 0.20).
- Frequent emotions like joy, love, and admiration were predicted reliably, while rare emotions like relief, pride, and grief were less accurate.  

## Future Improvements

- Enhance detection of rare or subtle emotions.
- Experiment with transformer-based models or deeper neural networks.
- Allow multi-label predictions for mixed emotions.
- Integrate multi-modal inputs such as voice tone or facial expressions for richer emotion analysis.

## Repository Structure
```
├─ docs/
│ ├─ notebook.ipynb
│ ├─ presentation.txt
│ └─ report.pdf
├─ data/
│ ├─ goemotions_1.csv
│ ├─ goemotions_2.csv
│ └─ goemotions_3.csv
├─ embeddings/
│ └─ glove.6B.100d.txt
└─ README.md
```

## Usage

1. Clone the repository
2. Preprocess the dataset using the provided notebook
3. Train and evaluate models
4. Use the demonstration interface to test new comments

## References

- Chollet, F. (2017). Deep learning with Python. Manning Publications.  
- Demszky, D., Movshovitz-Attias, D., Ko, J., Ravi, S., Fung, P., & Chang, M. (2020). GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://aclanthology.org/2020.acl-main.372  
- Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. In Proceedings of the European Conference on Machine Learning. https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf  
- Ma, F., Cai, Y., & Gao, J. (2020). Regularised text logistic regression: Key word detection and sentiment classification for online reviews. arXiv. https://arxiv.org/abs/2009.04591  
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to information retrieval. Cambridge University Press. https://nlp.stanford.edu/IR-book/  
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global vectors for word representation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). https://nlp.stanford.edu/projects/glove/  
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv. https://arxiv.org/abs/1609.04747

