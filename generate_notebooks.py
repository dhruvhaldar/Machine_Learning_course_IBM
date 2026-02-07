import os
import nbformat as nbf

def create_notebook(title, description, code_cells, filename):
    nb = nbf.v4.new_notebook()

    # Header
    header_md = f"""# {title}

{description}

## Objectives

*   Load and explore the dataset.
*   Visualize the data.
*   Train the model using Scikit-Learn.
*   Evaluate the model performance.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))

    # Imports
    imports_code = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline"""
    nb.cells.append(nbf.v4.new_code_cell(imports_code))

    # Add project specific cells
    for cell_type, content in code_cells:
        if cell_type == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        else:
            nb.cells.append(nbf.v4.new_code_cell(content))

    # Write to file
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

def main():
    projects = [
        {
            "name": "Multiple_Linear_Regression",
            "title": "Multiple Linear Regression",
            "desc": "Predicting housing prices using multiple features from the California Housing dataset.",
            "cells": [
                ('markdown', '## 1. Load Dataset\nWe will use the **California Housing** dataset.'),
                ('code', """from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())"""),
                ('markdown', '## 2. Exploratory Data Analysis\nLet\'s visualize the relationship between features.'),
                ('code', """plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()"""),
                ('markdown', '## 3. Train Model\nWe will split the data and train a Multiple Linear Regression model.'),
                ('code', """from sklearn.linear_model import LinearRegression

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')"""),
                ('markdown', '## 4. Evaluation\nEvaluate the model using Mean Squared Error and R2 Score.'),
                ('code', """y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()""")
            ]
        },
        {
            "name": "Polynomial_Regression",
            "title": "Polynomial Regression",
            "desc": "Modeling non-linear relationships using Polynomial Regression on synthetic data.",
            "cells": [
                ('markdown', '## 1. Generate Data\nWe generate non-linear synthetic data.'),
                ('code', """np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)
X = X[:, np.newaxis]

plt.scatter(X, y)
plt.title('Synthetic Non-Linear Data')
plt.show()"""),
                ('markdown', '## 2. Train Model\nTransform features to polynomial features and fit a Linear Regression model.'),
                ('code', """from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Transform to polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)"""),
                ('markdown', '## 3. Visualization\nPlot the polynomial regression curve.'),
                ('code', """import operator
plt.scatter(X, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
x_plot, y_poly_plot = zip(*sorted_zip)
plt.plot(x_plot, y_poly_plot, color='m')
plt.title('Polynomial Regression Fit')
plt.show()"""),
                ('markdown', '## 4. Evaluation'),
                ('code', """print('R2 Score:', metrics.r2_score(y, y_poly_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_poly_pred))""")
            ]
        },
        {
            "name": "Logistic_Regression",
            "title": "Logistic Regression",
            "desc": "Classification of Breast Cancer dataset using Logistic Regression.",
            "cells": [
                ('markdown', '## 1. Load Dataset\nWe use the **Breast Cancer Wisconsin** dataset.'),
                ('code', """from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df['target'].value_counts())"""),
                ('markdown', '## 2. Train Model'),
                ('code', """from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)"""),
                ('markdown', '## 3. Evaluation\nConfusion Matrix and Classification Report.'),
                ('code', """y_pred = model.predict(X_test)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))""")
            ]
        },
        {
            "name": "K_Nearest_Neighbors",
            "title": "K-Nearest Neighbors (KNN)",
            "desc": "Classification using KNN on the Iris dataset.",
            "cells": [
                ('markdown', '## 1. Load Dataset\nWe use the classic **Iris** dataset.'),
                ('code', """from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())"""),
                ('markdown', '## 2. Train Model\nWe need to choose K. Let\'s start with K=5.'),
                ('code', """from sklearn.neighbors import KNeighborsClassifier

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)"""),
                ('markdown', '## 3. Evaluation'),
                ('code', """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Finding optimal K
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()""")
            ]
        },
        {
            "name": "Decision_Trees",
            "title": "Decision Trees",
            "desc": "Classification using Decision Trees on the Wine dataset.",
            "cells": [
                ('markdown', '## 1. Load Dataset\nWe use the **Wine** dataset.'),
                ('code', """from sklearn.datasets import load_wine
data = load_wine()
X = data.data
y = data.target
print(f"Features: {data.feature_names}")"""),
                ('markdown', '## 2. Train Model'),
                ('code', """from sklearn.tree import DecisionTreeClassifier, plot_tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)"""),
                ('markdown', '## 3. Visualization and Evaluation'),
                ('code', """plt.figure(figsize=(15,10))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))""")
            ]
        },
        {
            "name": "Support_Vector_Machines",
            "title": "Support Vector Machines (SVM)",
            "desc": "Classification using SVM on the Breast Cancer dataset.",
            "cells": [
                ('markdown', '## 1. Load Dataset'),
                ('code', """from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target"""),
                ('markdown', '## 2. Train Model'),
                ('code', """from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear') # Try 'rbf' or 'poly' as well
model.fit(X_train, y_train)"""),
                ('markdown', '## 3. Evaluation'),
                ('code', """y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))""")
            ]
        },
        {
            "name": "Clustering_KMeans",
            "title": "Clustering with K-Means",
            "desc": "Unsupervised learning to cluster synthetic data.",
            "cells": [
                ('markdown', '## 1. Generate Data'),
                ('code', """from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()"""),
                ('markdown', '## 2. Train Model\nWe use the Elbow Method to find the optimal number of clusters.'),
                ('code', """from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()"""),
                ('markdown', '## 3. Visualize Clusters'),
                ('code', """kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=pred_y, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('Clusters with Centroids')
plt.show()""")
            ]
        },
        {
            "name": "Random_Forest",
            "title": "Random Forest",
            "desc": "Ensemble learning using Random Forest on the Digits dataset.",
            "cells": [
                ('markdown', '## 1. Load Dataset\nWe use the **Digits** dataset for handwritten digit recognition.'),
                ('code', """from sklearn.datasets import load_digits
data = load_digits()
X = data.data
y = data.target

plt.gray()
plt.matshow(data.images[0])
plt.show()"""),
                ('markdown', '## 2. Train Model'),
                ('code', """from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)"""),
                ('markdown', '## 3. Evaluation'),
                ('code', """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()""")
            ]
        }
    ]

    for project in projects:
        directory = project["name"]
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"{project['name']}.ipynb")
        create_notebook(project['title'], project['desc'], project['cells'], filename)

if __name__ == "__main__":
    main()
