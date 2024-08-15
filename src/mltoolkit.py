
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from itertools import cycle

class MLToolkit:
    def __init__(self, X, y, test_size=0.3, random_state=1):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.sc = StandardScaler()
        self.X_train_std = self.sc.fit_transform(self.X_train)
        self.X_test_std = self.sc.transform(self.X_test)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.n_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_multiclass_roc(self, y_true, y_score, title):
        y_true_bin = label_binarize(y_true, classes=self.classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def plot_feature_importance(self, model, X, y, title):
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        sorted_idx = result.importances_mean.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(range(X.shape[1]), result.importances_mean[sorted_idx])
        plt.yticks(range(X.shape[1]), [f'Feature {i}' for i in sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, model, X_test, y_test, y_pred, y_pred_proba=None):
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cv_scores = cross_val_score(model, self.X_train_std, self.y_train, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        self.plot_confusion_matrix(y_test, y_pred, f'{model.__class__.__name__} Confusion Matrix')
        
        if y_pred_proba is not None:
            self.plot_multiclass_roc(y_test, y_pred_proba, f'{model.__class__.__name__} Multiclass ROC Curve')

        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            self.plot_feature_importance(model, self.X_train_std, self.y_train, f'{model.__class__.__name__} Feature Importance')

    def logistic_regression(self, C=1.0, penalty='l2', solver='lbfgs'):
        print(f"\nLogistic Regression (C={C}, penalty={penalty}):")
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=1)
        lr.fit(self.X_train_std, self.y_train)
        y_pred = lr.predict(self.X_test_std)
        y_pred_proba = lr.predict_proba(self.X_test_std)
        self.evaluate_model(lr, self.X_test_std, self.y_test, y_pred, y_pred_proba)
        return lr

    def svm(self, kernel='rbf', C=1.0):
        print(f"\nSupport Vector Machine (kernel={kernel}, C={C}):")
        svm = SVC(kernel=kernel, C=C, probability=True, random_state=1)
        svm.fit(self.X_train_std, self.y_train)
        y_pred = svm.predict(self.X_test_std)
        y_pred_proba = svm.predict_proba(self.X_test_std)
        self.evaluate_model(svm, self.X_test_std, self.y_test, y_pred, y_pred_proba)
        return svm

    def decision_tree(self, criterion='gini', max_depth=None):
        print(f"\nDecision Tree (criterion={criterion}, max_depth={max_depth}):")
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=1)
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        self.evaluate_model(tree, self.X_test, self.y_test, y_pred)
        return tree

    def random_forest(self, n_estimators=100, max_depth=None):
        print(f"\nRandom Forest (n_estimators={n_estimators}, max_depth={max_depth}):")
        forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
        forest.fit(self.X_train, self.y_train)
        y_pred = forest.predict(self.X_test)
        self.evaluate_model(forest, self.X_test, self.y_test, y_pred)
        return forest

    def knn(self, n_neighbors=5, p=2, metric='minkowski'):
        print(f"\nK-Nearest Neighbors (n_neighbors={n_neighbors}, p={p}, metric={metric}):")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric)
        knn.fit(self.X_train_std, self.y_train)
        y_pred = knn.predict(self.X_test_std)
        self.evaluate_model(knn, self.X_test_std, self.y_test, y_pred)
        return knn

# Example usage:
# if __name__ == "__main__":
#     # Load the iris dataset
#     iris = datasets.load_iris()
#     X = iris.data[:, [2, 3]]
#     y = iris.target

#     # Create an instance of the MLToolkit
#     ml_toolkit = MLToolkit(X, y)

#     # Run different algorithms
#     lr = ml_toolkit.logistic_regression(C=100.0)
#     lr_l2 = ml_toolkit.logistic_regression(C=0.01, penalty='l2')
#     lr_l1 = ml_toolkit.logistic_regression(C=0.01, penalty='l1', solver='liblinear')
#     svm = ml_toolkit.svm(kernel='linear', C=1.0)
#     tree = ml_toolkit.decision_tree(criterion='gini', max_depth=4)
#     forest = ml_toolkit.random_forest(n_estimators=100)
#     knn = ml_toolkit.knn(n_neighbors=5)
