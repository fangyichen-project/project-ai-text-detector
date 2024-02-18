# project-ai-text-detector
A project to do classification for llm generated texts and human written texts in German.
The goal is to compare the results using three different classification models random forest, svm and word embedding.
```mermaid
  flowchart TD
    A[Human wrtten texts from Newswebsite A] -->|Scrapy Spider| B(Pymongo DB Collection Human Written Texts)
    C[Human wrtten texts from Newswebsite B]-->|Scrapy Spider| B(Pymongo DB Collection Human Written Texts)
    B-->|Text Cleaning|D[Cleaned Human wrtten texts]
    D-->|Text Preprocessing|E[Stemmatized human written texts without stopwords]
    F[pre-existed cleaned LLM generated texts] -->|Text Preprocessing|G[Stemmatized LLM generated texts without stopwords]
    G-->H[Data Set in Mongo DB]
    E-->H[Data Set in Mongo DB]
    H-->|Split in training and testing datasets|I[X train, X test, y train, y test]
    I-->|Vectorizing using TF-IDF|J[TF-IDF Matrix]
    J-->|Random Forest|K[Prediction Rate using RF]
    J-->|Support Vector Machine|L[Prediction Rate using svm]
    I-->|Word Embedding|M[Prediction Rate using word embedding]
```
