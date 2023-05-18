from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


class Classifier:
    # Creating a database
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

    # Split database into training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier object and fit it to the training data
    tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    tree_entropy.fit(X_train, y_train)

    tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    tree_gini.fit(X_train, y_train)

    # Investigate the decision tree classifier for different depths
    tree_entropy_2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42)
    tree_entropy_2.fit(X_train, y_train)

    tree_gini_4 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    tree_gini_4.fit(X_train, y_train)

    # Create a random forest classifier object and fit it to the training data
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)

    # Test the performance of the classifier for different numbers of decision trees
    forest_50 = RandomForestClassifier(n_estimators=50, random_state=42)
    forest_50.fit(X_train, y_train)

    forest_200 = RandomForestClassifier(n_estimators=200, random_state=42)
    forest_200.fit(X_train, y_train)

    # Train logistic regression and SVM classifiers
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    svm = SVC(kernel='linear', C=0.025, random_state=42)
    svm.fit(X_train, y_train)

    # Combine the SVM, logistic regression, and random forest classifiers into one group
    voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('forest', forest)], voting='hard')
    voting_clf.fit(X_train, y_train)

    # Predictions for decision tree classifiers
    y_pred_entropy = tree_entropy.predict(X_test)
    y_pred_gini = tree_gini.predict(X_test)
    y_pred_entropy_2 = tree_entropy_2.predict(X_test)
    y_pred_gini_4 = tree_gini_4.predict(X_test)

    # Predictions for random forest classifiers
    y_pred_forest = forest.predict(X_test)
    y_pred_forest_50 = forest_50.predict(X_test)
    y_pred_forest_200 = forest_200.predict(X_test)

    # Predictions for logistic regression and SVM classifiers
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_svm = svm.predict(X_test)

    # Predictions for voting classifier
    y_pred_voting = voting_clf.predict(X_test)

    # Accuracy scores for all classifiers
    print("Accuracy score for decision tree (entropy):", accuracy_score(y_test, y_pred_entropy))
    print("Accuracy score for decision tree (gini):", accuracy_score(y_test, y_pred_gini))
    print("Accuracy score for decision tree (entropy, max_depth=2):", accuracy_score(y_test, y_pred_entropy_2))
    print("Accuracy score for decision tree (gini, max_depth=4):", accuracy_score(y_test, y_pred_gini_4))
    print("Accuracy score for random forest (n_estimators=100):", accuracy_score(y_test, y_pred_forest))
    print("Accuracy score for random forest (n_estimators=50):", accuracy_score(y_test, y_pred_forest_50))
    print("Accuracy score for random forest (n_estimators=200):", accuracy_score(y_test, y_pred_forest_200))
    print("Accuracy score for logistic regression:", accuracy_score(y_test, y_pred_log_reg))
    print("Accuracy score for SVM:", accuracy_score(y_test, y_pred_svm))
    print("Accuracy score for voting classifier:", accuracy_score(y_test, y_pred_voting))

    # Accuracy scores for all classifiers
    scores = [accuracy_score(y_test, y_pred_entropy), accuracy_score(y_test, y_pred_gini),
              accuracy_score(y_test, y_pred_entropy_2), accuracy_score(y_test, y_pred_gini_4),
              accuracy_score(y_test, y_pred_forest), accuracy_score(y_test, y_pred_forest_50),
              accuracy_score(y_test, y_pred_forest_200), accuracy_score(y_test, y_pred_log_reg),
              accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_voting)]

    # Names of the classifiers
    names = ['Entropy', 'Gini', 'Entropy (max_depth=2)', 'Gini (max_depth=4)', 'Random Forest (100 trees)',
             'Random Forest (50 trees)', 'Random Forest (200 trees)', 'Logistic Regression', 'SVM', 'Voting Classifier']

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(names, scores, color='blue')
    plt.title('Accuracy Scores for Classification Algorithms')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy Score')
    plt.ylim((0.85, 0.9))
    plt.xticks(rotation=45, ha='right')
    plt.show()
