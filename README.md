# project-ai-text-detector (On Working)
A project to do classification for llm generated texts and human written texts in German language.
The goal is to compare the results using different classification models (including random forest and svm).
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
    J-->|Classification Tree|K[Prediction Rate using Classification Tree] 
    J-->|Support Vector Machine|L[Prediction Rate using svm]
    I-->|Word Embedding|M[(TODO)] 
```
