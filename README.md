# Cyberbullying-NLP 
[![DOI](https://zenodo.org/badge/268175859.svg)](https://zenodo.org/badge/latestdoi/268175859)

## How to run the pipeline 

`ipython --pdb pipeline.py ./DS2_clean.csv` 

Results will show on the screen like the following: 

```
NaiveBayes with Tfidf
GaussianNB(priors=None, var_smoothing=1e-09)
              precision    recall  f1-score   support

           0       0.31      0.95      0.47      1234
           1       0.98      0.59      0.73      6201

    accuracy                           0.65      7435
   macro avg       0.65      0.77      0.60      7435
weighted avg       0.87      0.65      0.69      7435

-----------------------------------

NaiveBayes with Word2Vec-TFIDF
GaussianNB(priors=None, var_smoothing=1e-09)
              precision    recall  f1-score   support

           0       0.26      0.72      0.38      1234
           1       0.92      0.59      0.72      6201

    accuracy                           0.61      7435
   macro avg       0.59      0.66      0.55      7435
weighted avg       0.81      0.61      0.66      7435

-----------------------------------

NaiveBayes with Word2Vec
GaussianNB(priors=None, var_smoothing=1e-09)
              precision    recall  f1-score   support

           0       0.65      0.83      0.73      1234
           1       0.96      0.91      0.94      6201

    accuracy                           0.90      7435
   macro avg       0.81      0.87      0.83      7435
weighted avg       0.91      0.90      0.90      7435

-----------------------------------

DecisionTree with Tfidf
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
              precision    recall  f1-score   support

           0       0.77      0.83      0.80      1234
           1       0.97      0.95      0.96      6201

    accuracy                           0.93      7435
   macro avg       0.87      0.89      0.88      7435
weighted avg       0.93      0.93      0.93      7435

-----------------------------------

DecisionTree with Word2Vec-TFIDF
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
              precision    recall  f1-score   support

           0       0.30      0.33      0.31      1234
           1       0.86      0.84      0.85      6201

    accuracy                           0.76      7435
   macro avg       0.58      0.59      0.58      7435
weighted avg       0.77      0.76      0.76      7435

-----------------------------------

DecisionTree with Word2Vec
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
              precision    recall  f1-score   support

           0       0.63      0.65      0.64      1234
           1       0.93      0.92      0.93      6201

    accuracy                           0.88      7435
   macro avg       0.78      0.79      0.78      7435
weighted avg       0.88      0.88      0.88      7435

-----------------------------------

NaiveBayes with FastText
GaussianNB(priors=None, var_smoothing=1e-09)
              precision    recall  f1-score   support

           0       0.55      0.73      0.63      1234
           1       0.94      0.88      0.91      6201

    accuracy                           0.86      7435
   macro avg       0.75      0.81      0.77      7435
weighted avg       0.88      0.86      0.87      7435

-----------------------------------

RandomForest with Tfidf
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
              precision    recall  f1-score   support

           0       0.80      0.87      0.83      1234
           1       0.97      0.96      0.96      6201

    accuracy                           0.94      7435
   macro avg       0.89      0.91      0.90      7435
weighted avg       0.94      0.94      0.94      7435

-----------------------------------

AdaBoost with Tfidf
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
              precision    recall  f1-score   support

           0       0.76      0.95      0.85      1213
           1       0.99      0.94      0.97      6222

    accuracy                           0.94      7435
   macro avg       0.88      0.95      0.91      7435
weighted avg       0.95      0.94      0.95      7435

-----------------------------------

SVC with Word2Vec
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
              precision    recall  f1-score   support

           0       0.80      0.77      0.79      1234
           1       0.96      0.96      0.96      6201

    accuracy                           0.93      7435
   macro avg       0.88      0.87      0.87      7435
weighted avg       0.93      0.93      0.93      7435

-----------------------------------

RandomForest with Word2Vec-TFIDF
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
              precision    recall  f1-score   support

           0       0.63      0.11      0.19      1234
           1       0.85      0.99      0.91      6201

    accuracy                           0.84      7435
   macro avg       0.74      0.55      0.55      7435
weighted avg       0.81      0.84      0.79      7435

-----------------------------------

RandomForest with Word2Vec
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
              precision    recall  f1-score   support

           0       0.83      0.64      0.72      1234
           1       0.93      0.97      0.95      6201

    accuracy                           0.92      7435
   macro avg       0.88      0.81      0.84      7435
weighted avg       0.92      0.92      0.91      7435

-----------------------------------

AdaBoost with Word2Vec
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
              precision    recall  f1-score   support

           0       0.76      0.68      0.72      1213
           1       0.94      0.96      0.95      6222

    accuracy                           0.91      7435
   macro avg       0.85      0.82      0.83      7435
weighted avg       0.91      0.91      0.91      7435

-----------------------------------

AdaBoost with Word2Vec-TFIDF
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
              precision    recall  f1-score   support

           0       0.47      0.15      0.23      1213
           1       0.85      0.97      0.91      6222

    accuracy                           0.83      7435
   macro avg       0.66      0.56      0.57      7435
weighted avg       0.79      0.83      0.80      7435

-----------------------------------

SVC with Word2Vec-TFIDF
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
              precision    recall  f1-score   support

           0       0.67      0.00      0.00      1234
           1       0.83      1.00      0.91      6201

    accuracy                           0.83      7435
   macro avg       0.75      0.50      0.46      7435
weighted avg       0.81      0.83      0.76      7435

-----------------------------------

DecisionTree with FastText
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
              precision    recall  f1-score   support

           0       0.51      0.53      0.52      1234
           1       0.91      0.90      0.90      6201

    accuracy                           0.84      7435
   macro avg       0.71      0.71      0.71      7435
weighted avg       0.84      0.84      0.84      7435

-----------------------------------

SVC with FastText
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
              precision    recall  f1-score   support

           0       0.79      0.67      0.73      1234
           1       0.94      0.96      0.95      6201

    accuracy                           0.92      7435
   macro avg       0.86      0.82      0.84      7435
weighted avg       0.91      0.92      0.91      7435

-----------------------------------

AdaBoost with FastText
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
              precision    recall  f1-score   support

           0       0.74      0.60      0.66      1253
           1       0.92      0.96      0.94      6182

    accuracy                           0.90      7435
   macro avg       0.83      0.78      0.80      7435
weighted avg       0.89      0.90      0.89      7435

-----------------------------------

RandomForest with FastText
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
              precision    recall  f1-score   support

           0       0.85      0.49      0.62      1234
           1       0.91      0.98      0.94      6201

    accuracy                           0.90      7435
   macro avg       0.88      0.74      0.78      7435
weighted avg       0.90      0.90      0.89      7435

-----------------------------------

MLPClassifier with Word2Vec
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
              precision    recall  f1-score   support

           0       0.82      0.72      0.77      1213
           1       0.95      0.97      0.96      6222

    accuracy                           0.93      7435
   macro avg       0.88      0.85      0.86      7435
weighted avg       0.93      0.93      0.93      7435

-----------------------------------

MLPClassifier with Word2Vec-TFIDF
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
              precision    recall  f1-score   support

           0       0.53      0.33      0.41      1213
           1       0.88      0.94      0.91      6222

    accuracy                           0.84      7435
   macro avg       0.70      0.64      0.66      7435
weighted avg       0.82      0.84      0.83      7435

-----------------------------------

MLPClassifier with FastText
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
              precision    recall  f1-score   support

           0       0.78      0.73      0.75      1246
           1       0.95      0.96      0.95      6189

    accuracy                           0.92      7435
   macro avg       0.86      0.84      0.85      7435
weighted avg       0.92      0.92      0.92      7435

-----------------------------------

SVC with Tfidf
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
              precision    recall  f1-score   support

           0       0.80      0.88      0.84      1234
           1       0.98      0.96      0.97      6201

    accuracy                           0.94      7435
   macro avg       0.89      0.92      0.90      7435
weighted avg       0.95      0.94      0.94      7435

-----------------------------------

MLPClassifier with Tfidf
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
              precision    recall  f1-score   support

           0       0.82      0.82      0.82      1246
           1       0.96      0.96      0.96      6189

    accuracy                           0.94      7435
   macro avg       0.89      0.89      0.89      7435
weighted avg       0.94      0.94      0.94      7435

-----------------------------------

GradientBoosting with Tfidf
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=5,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
              precision    recall  f1-score   support

           0       0.84      0.59      0.69      1234
           1       0.92      0.98      0.95      6201

    accuracy                           0.91      7435
   macro avg       0.88      0.78      0.82      7435
weighted avg       0.91      0.91      0.91      7435

-----------------------------------

GradientBoosting with Word2Vec
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
              precision    recall  f1-score   support

           0       0.81      0.70      0.75      1234
           1       0.94      0.97      0.95      6201

    accuracy                           0.92      7435
   macro avg       0.87      0.83      0.85      7435
weighted avg       0.92      0.92      0.92      7435

-----------------------------------

GradientBoosting with Word2Vec-TFIDF
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=5,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
              precision    recall  f1-score   support

           0       0.61      0.11      0.18      1234
           1       0.85      0.99      0.91      6201

    accuracy                           0.84      7435
   macro avg       0.73      0.55      0.55      7435
weighted avg       0.81      0.84      0.79      7435

-----------------------------------

GradientBoosting with FastText
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=5,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
              precision    recall  f1-score   support

           0       0.80      0.55      0.65      1234
           1       0.92      0.97      0.94      6201

    accuracy                           0.90      7435
   macro avg       0.86      0.76      0.80      7435
weighted avg       0.90      0.90      0.89      7435

-----------------------------------
``` 


## TODO 

### Word Embedding 

- [ ] Elmo 
- [x] Word2Vec 
- [x] TFIDF 
- [x] Word2vec+TFIDF 
- [x] FastText 
- [ ] BERT 
- [ ] ULM-Fit 
- [ ] GloVe 

## Classifier 
- [x] SVM (kernel: linear) 
- [x] Perceptron 
- [x] Decision Tree 
- [x] AdaBoost 
- [x] Random Forest 
- [x] Gradient Boosting 
- [x] Gusassion NB 
- [x] Multinomial NB 

