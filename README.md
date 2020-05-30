# Cyberbullying-NLP 
conference paper 

## How to run the pipeline 

`ipython --pdb tfidf.py` 

Results will show on the screen like the following: 

```
MultinomialNB Classifier Started!
NaiveBayes Classifier Started!
SVC Classifier Started!
DecisionTree Classifier Started!
Perceptron Classifier Started!
RandomForest Classifier Started!
GradientBoosting Classifier Started!
LogisticRegression Classifier Started!
AdaBoost Classifier Started!
MLPClassifier Classifier Started!
NaiveBayes : {}
Accuracy: 0.543         Precision: 0.555        Recall: 0.597           F1: 0.494

MultinomialNB : {'alpha': [0.5, 1], 'fit_prior': [True, False]}
Accuracy: 0.907         Precision: 0.841        Recall: 0.820           F1: 0.830

LogisticRegression : {}
Accuracy: 0.928         Precision: 0.903        Recall: 0.831           F1: 0.861

DecisionTree : {'min_samples_split': [2, 5]}
Accuracy: 0.930         Precision: 0.878        Recall: 0.875           F1: 0.876

RandomForest : {}
Accuracy: 0.934         Precision: 0.896        Recall: 0.862           F1: 0.878

AdaBoost : {}
Accuracy: 0.911         Precision: 0.887        Recall: 0.779           F1: 0.819

``` 
