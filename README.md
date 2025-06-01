# Child-Friendly-Shows
Deployment of 3 models to compare accuracy in predicting child-friendliness of movies and TV shows using IMDb dataset
Dataset: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
1) Introduction:
With the growing availability of digital content across streaming platforms, it has become increasingly important to identify and classify media that is suitable for children. Parents, educators, and platforms alike need reliable tools to filter out inappropriate content and recommend age-appropriate shows and movies.

This project focuses on building a machine learning model to predict whether a movie or TV show is child-friendly using publicly available metadata. The dataset includes attributes such as title, release year, certification, runtime, genre, IMDB rating, meta score, director, cast, vote count, and gross earnings.

Our objective is to engineer a target variable indicating child-friendliness and apply various supervised learning models to evaluate their performance. Rather than relying on a single model, we evaluated multiple machine learning algorithms - Logistic Regression, Decision Tree and Random Forest to determine the most effective approach to determine suitability of TV/OTT show for kids. This comparative analysis ensures that our final choice is justified through performance metrics.

2) Data Collection and Processing:
The dataset contains the following records: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'

Preprocessing Steps:

Removing columns with missing values
Label encoding and one-hot encoding for categorical variables
Feature scaling for numeric columns
Removing irrelevant columns like Poster_Link, Overview, Series_title
New column: 'is_child_friendly' created based on 'Certificate' as a target column
Dropped irrelevant and non-informative features
Categorical variables were one-hot encoded
The data was split into training and testing sets (80-20)
The data was split into training and testing sets using train_test_split (80:20)
3) Visualization:
A bar graph showed the mean MoCA score by diagnosis category
A scatter plot of UPDRS vs MoCA revealed the correlation patterns between the two
A line graph displayed the trend of scores for the first 50 patients
These visualizations provided insights into data imbalance, potential feature significance, and correlation between symptoms.

Visualization for decision tree and logistic regression was obtained as follows:

4) Feature Selection:
To make the model more accurate and efficient, we selected only the most useful features.

We started with all available features, like 'Poster_Link', 'Series_Title', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Certificate'.

We standardized the numerical features and converted categorical ones using one-hot encoding.

Feature importance scores were extracted using Random Forests and Decision Trees after model training to interpret the contribution of each feature. While these insights were not used to retrain models with reduced features in this version, they provide a valuable direction for future optimization.

Top features found:

Certificate (Most critical - explicitly states age appropriateness)
Genre (Family-friendly genres are strong indicators)
Overview (NLP can extract themes/content warnings)
Meta_score (Critical reception may correlate with appropriateness)
Runtime (Very short/long durations may correlate with target demographics)
Additional top features:

Cognitive ability measure is strongly associated with diagnosis
UPDRS: Clinical indicator of Parkinson's symptoms
Age: Higher risk is associated with aging
Education Level and Ethnicity also showed moderate influence
This helped us in prioritizing features that really mattered for prediction.

5) Model Selection:
We tried three different models to find the best one for predicting Child-friendly TV/OTT shows:

1. Logistic Regression: A linear model that is simple, fast, and interpretable. It provides a solid baseline for binary classification problems.

2. Gaussian Naive Bayes: A probabilistic model that works well on small datasets but may oversimplify complex patterns.

3. Decision Tree Classifier: Captures non-linear relationships and is easy to visualize. However, it is prone to overfitting if not properly tuned.

4. Random Forest Classifier: It is an ensemble of trees, i.e. it uses multiple decision trees to make decisions. This leads to a significant improvement in accuracy and other evaluation metrics. Thus, the model is robust and less prone to bias.

Why Random Forest performed best:

It achieved the highest accuracy on the test data
It handled the dataset's structure (categorical as well as numeric values) and variance effectively
It showed better generalization and lower variance compared to a single decision tree
It also provided feature importance scores, helping us understand which symptoms had the most predictive power
We compared each model using accuracy, confusion matrix, and precision/recall scores. Random Forest gave the best overall results.

6) Model Description / Algorithm:
Each model uses a different method to learn from the data and make predictions.

a. Logistic Regression

Logistic Regression is a linear classification algorithm used for binary outcomes (Suitable/Unsuitable for kids)
It calculates the probability of a show being kid-friendly using a mathematical function called the logistic (sigmoid) function, which outputs values between 0 and 1
c. Decision Tree Classifier

This model uses a tree-like structure where nodes split the data based on feature values that maximize information gain (using metrics like Gini Impurity or Entropy)
The tree continues splitting until stopping criteria are met (e.g., max depth or minimum samples per leaf)
It can capture non-linear relationships and is easy to visualize
d. Random Forest Classifier Random Forest is a robust ensemble learning algorithm that combines the outputs of multiple decision trees to improve classification accuracy and reduce overfitting. Instead of relying on a single decision tree, which may be prone to bias or variance, Random Forest builds a "forest" of diverse trees and aggregates their results for more reliable predictions.

Some more information:

Features used: IMDB_Rating and Meta_score
Train-Test Split: 80/20
Label: Certificate mapped to child suitability (U, PG → 1)
Following Scatterplot was obtained.

Algorithm:

Data Loading and Preparation

Unzip and load the dataset from 'imdb_top_1000.csv'
Examine the columns in the dataset
Remove unnecessary columns: 'Poster_Link', 'Series_title', 'Overview'
Handle missing values by:
First filling with 0
Then filling remaining missing values with "Unknown"
Feature Engineering

Create a binary target variable 'Is_child_friendly':
Define child-friendly certificates: ['U', 'UA', 'G', 'TV-Y', 'TV-G', 'PG', 'TV-Y?']
Apply function to 'Certificate' column to create binary indicator (1=child-friendly, 0=not)
Data Preprocessing

Encode categorical features (not shown in code but implied by context)
Split data into features (X) and target (y)
Split dataset into training and testing sets (implied by later steps)
Model Training

Initialize Random Forest Classifier with:
n_estimators=100
random_state=42
Fit the model on training data (X_train, y_train)
Model Evaluation

Make predictions on test set (X_test)
Calculate accuracy score by comparing predictions (y_pred) with true values (y_test)
Generate classification report showing:
Precision, recall, f1-score for both classes
Support count for each class
Macro and weighted averages
Visualization (Optional)

Select first tree from the Random Forest
Plot the decision tree with:
Feature names
Class names ('Not Child-Friendly', 'Child-Friendly')
Filled colors for visualization
Output:

Accuracy score: 0.855
Classification report with performance metrics
Decision tree visualization (if executed)
7) Testing and Evaluation of Models
After training the machine learning models, we evaluated their performance using the test set. This helps us assess how well each model generalizes to new, unseen patient data.

Evaluation Metrics Used:

Accuracy: Percentage of total predictions that were correct. Accuracy = Correct Predictions/Total Predictions

Confusion Matrix: Provides a detailed breakdown of the model's predictions

Classification Report:
Precision = TP/(TP+FP) – How many predicted "Child-Friendly" shows were actually child-friendly
Recall = TP/(TP+FN) – How many "Child-Friendly" were correctly detected
F1-score – Evaluates the model's performance by balancing precision and recall. F1-score = (2 × Precision × Recall) / (Precision + Recall)
Higher F1-score indicates a balanced and reliable model.

After training and testing all 3 models, we compared their performance to find the most effective one for determining Child-Friendly TV/OTT shows.

8) Results

Key Outcomes:
The Random Forest Classifier had the highest accuracy among all models
It also showed strong recall and precision, meaning it was good at correctly identifying child-friendly shows and minimizing false positives
Random Forest achieved the highest accuracy (85.5%) and F1-score (84%), making it the most reliable classifier
Decision Tree also showed performance, but had lower predictive power as compared to Random Forest
Logistic Regression also performed well, but was found to be the most important features
'Runtime', 'Genre' and 'IMDB_Rating' were found to be the most important features influencing the diagnosis
The model was able to handle both numerical and categorical data effectively due to proper preprocessing
9) Conclusion:
In this project, we aimed to predict whether a movie or TV show is child-friendly based on various features such as genre, runtime, rating, cast, and more.

We engineered a binary target variable 'is_child_friendly' by analyzing the certification labels, and prepared the dataset through appropriate data cleaning, transformation, and encoding techniques.

We then implemented and evaluated three different machine learning models:

Decision Tree Classifier: Accuracy – 0.626
Logistic Regression: Accuracy – 0.59
Random Forest Classifier: Accuracy – 0.855
Among the models tested, the Random Forest Classifier significantly outperformed the others, achieving the highest accuracy of 85.5%. This confirms that ensemble methods like Random Forest, which combine multiple decision trees, are more robust and effective in handling diverse and non-linear feature interactions compared to simpler models.

Overall, this model can serve as a helpful tool in identifying child-appropriate content with high reliability, and can be extended further with more advanced NLP and content metadata analysis for even better performance.





