# Report on Classification Model for Bank Marketing Data

## 1. Data Overview

The dataset contains bank client data related to a marketing campaign. This data includes demographic information like age, job, marital status, and education level, as well as financial information like whether the client has a housing loan, personal loan, or credit in default. The data also contains details about the campaign, such as the communication type, month and day of the week of the contact, and the campaign's duration.

## 2. Data Processing

Before applying any machine learning models, the data was preprocessed. This involved:

- Converting categorical variables into a format suitable for machine learning models using one-hot encoding for nominal variables and ordinal encoding for ordinal variables.
- Scaling the numerical features to have a mean of 0 and a variance of 1 using StandardScaler to ensure that the range of the features does not negatively impact the performance of certain models.

## 3. Model Selection and Hyperparameter Tuning

We performed a grid search over several models to identify the best one according to the F1 score. The models explored were SVM, k-Nearest Neighbors, and Decision Trees. For each model, a range of hyperparameters was considered. We also used StratifiedShuffleSplit for cross-validation to maintain the same class proportion across folds.

## 4. Evaluation Metrics 

The primary metric used to evaluate the models was the F1 score. This is a balanced measure of precision (how many of the positive predictions were correct) and recall (how many of the actual positive instances were identified). In addition to the F1 score, we also calculated accuracy (the proportion of correct predictions) and plotted ROC curves.

## 5. Data Loading and Visualization

The dataset was loaded from the 'bank-additional-full.csv' file, and an initial exploratory data analysis was performed. A heatmap of the correlation matrix was created to visualize the correlation between all features.

## 6. Feature Selection

Features of interest were selected for our predictive models, which included 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', and 'y'.

## 7. Data Preprocessing

A ColumnTransformer was used for data preprocessing, which combined StandardScaler, OrdinalEncoder, and OneHotEncoder to standardize, encode, and one-hot encode our data, respectively. After a train-test split, the data was fit to the preprocessor.

## 8. Modeling and Evaluation

- A baseline model was created using the DummyClassifier, and its accuracy score was recorded.
- Logistic Regression was implemented, which achieved a similar accuracy score as the baseline model.
- The K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machines (SVM) models were also tested, and their parameters, training times, and accuracies on the training and test sets were recorded and summarized in a DataFrame.

## 9. Stratified Sampling

Stratified sampling was performed to create a smaller training dataset for further analysis. The dimensions of the original and smaller datasets were compared.

## 10. Hyperparameter Tuning

Grid search was utilized for hyperparameter tuning on the selected models. The best parameters and scores for the f1 and accuracy metrics were identified for each model. If available, the top 10 features by importance were printed.

## 11. Plotting and Visualization

The ROC curve, Precision-Recall curve, and confusion matrix were plotted. If possible, a graph displaying the top 10 features by importance was also shown.

## 12. Further Analysis with Expanded Dataset

The analysis was repeated using an expanded dataset that included 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'month', 'day_of_week', and 'contact' as additional features.


## 13. Results

    The baseline model had an accuracy of 89%. 
    The logistic regression model had a similar accuracy of
    89%.

    Train time, train accuracy and test accuracy for initial set of features ('age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y') and default hyperparameters

    | Model         | Train Time | Train Accuracy | Test Accuracy |
    | ------------- | ---------- | -------------- | ------------- |
    | KNN           | 0.024195   | 0.889833       | 0.872420      |
    | DT            | 0.072749   | 0.916601       | 0.862224      |
    | SVM           | 12.878485  | 0.887557       | 0.886502      |

    Train time, train accuracy and test accuracy for initial set of features ('age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y') and grid search for optimal parameters

    | Model         | Train Time | Train Accuracy | Test Accuracy |
    | ------------- | ---------- | -------------- | ------------- |
    | KNN           | 0.024195   | 0.9097         | 0.8522        |
    | DT - Extended | 0.072749   | 0.9163         | 0.8656        |
    | SVM           | 176.14     | 0.8874         | 0.8874        |


    Model: KNN
    Best Estimator: KNeighborsClassifier(n_neighbors=3, weights='distance')
    Best Parameters: {'n_neighbors': 3, 'weights': 'distance'}

    Model: Decision Tree
    Best Estimator: DecisionTreeClassifier(max_depth=20, random_state=42)
    Best Parameters: {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 1}
    Top features for Decision Tree by importance:
        age: 0.3660013707554424
        job: 0.19010003283392005
        education: 0.1375453029192563
        housing_no: 0.0501605694285253

    Model: SVM
    Best Estimator: SVC(C=0.1, kernel='linear', probability=True, random_state=42)
    Best Parameters: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear', 'probability': True}
    Top features for SVM by importance:
        default_no: 0.0001148557973291442
        housing_yes: 9.56348937013185e-05
        age: 9.372477323424815e-05
        marital_single: 9.068895796746079e-05
        loan_no: 5.258992891407632e-05

    ------------------------------------------------------------------------------------------------------


    Train time, train accuracy and test accuracy for expanded set of features and grid search for optimal parameters

    | Model         | Train Time | Train Accuracy | Test Accuracy |
    | ------------- | ---------- | -------------- | ------------- |
    | KNN           | 17.8140    | 1.0000         | 0.8982        |
    | DT            | 0.072749   | 0.9319         | 0.9038        |
    | SVM           | 12.878485  | 0.9036         | 0.9002        |


    Grid search for KNN done
    Best parameters: {'n_neighbors': 5, 'weights': 'distance'}
    Best f1 score: 0.4327


    Model: Decision Tree
    Best parameters: {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 1}
    Best f1 score: 0.5112
    Top 10 features for Decision Tree by importance:
        duration: 0.4469679864601734
        pdays: 0.23016583059787077
        age: 0.06205527717266181
        month_mar: 0.035620884003431615
        month_oct: 0.030731128218811396
        month_jun: 0.01949971469483294
        campaign: 0.01900958911992089
        job: 0.018427501576849765

    Model: SVM
    Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'linear', 'probability': True}
    Best f1 score: 0.3833
    Best train accuracy: 0.9036
    Best test accuracy: 0.9002
    Top 10 features for SVM by importance:
        duration: 0.3664750851648404
        month_mar: 0.26948534426723114
        contact_cellular: 0.19295671689970817
        month_sep: 0.19219616286093189
        month_oct: 0.1548628827519849
        month_jun: 0.11586076919824251



## 13. Conclusions and Business Insights

All the models performed well with decision trees performing very slight better than SVM. This can be attributed to the much smaller sample size of the SVM that was used to reduce run time.

The top features seem to be 

    Call Duration
    Month of Contact
    Age

The bank can use these insights to target individuals who have a greater chance of subscribing to the deposits.


## 15. Future Work 

Moving forward, we could consider including additional models in our grid search, such as random forests or other ensemble methods. We could also experiment with different ways of encoding our categorical variables, or creating new features. Another possibility would be to gather more data, if available, as this could potentially improve our model's performance.

[NoteBook File](prompt_III.ipynb)
[Data File](bank-additional-full.csv)
