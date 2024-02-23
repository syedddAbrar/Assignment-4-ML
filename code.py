import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load ing the dataset
df = pd.read_csv(r"C:\\Users\\abrar\\OneDrive\\Desktop\\ml\\resume_data.csv")

# A1: Evaluating the  intraclass spread and interclass distances
numeric_columns = df.select_dtypes(include=np.number).columns # Here, we are selecting the numerical columns for the calculation of mean and std

# finding mean for each class
mean_class1 = np.mean(df[df['Category'] == 'Years Of Experience'][numeric_columns], axis=0)
mean_class2 = np.mean(df[df['Category'] == 'Age'][numeric_columns], axis=0)

# finding standard deviation for each class
std_class1 = np.std(df[df['Category'] == 'Years Of Experience'][numeric_columns], axis=0)
std_class2 = np.std(df[df['Category'] == 'Age'][numeric_columns], axis=0)

# finding the distance between means from mean_class1 and mean_class2
distance_between_means = np.linalg.norm(mean_class1 - mean_class2)

# A2: plotting the Density pattern for a feature using histogram
feature_to_plot = 'Age'
plt.hist(df[feature_to_plot], bins=10, color='blue', edgecolor='black')
plt.xlabel(feature_to_plot)
plt.ylabel('Frequency')
plt.title('Histogram for ' + feature_to_plot)
plt.show()

mean_feature = np.mean(df[feature_to_plot])
variance_feature = np.var(df[feature_to_plot])

# A3: finding the  Minkowski distance with varying r from 1 to 10
# converting the feature vectors to numeric type
feature_vector1 = pd.to_numeric(df.iloc[0].drop('Category'), errors='coerce')
feature_vector2 = pd.to_numeric(df.iloc[1].drop('Category'), errors='coerce')

# removing the  NaN values (if any is present) after conversion
feature_vector1.dropna(inplace=True)
feature_vector2.dropna(inplace=True)

# using the Minkowski distance calculation
r_values = range(1, 11)
minkowski_distances = [np.linalg.norm(feature_vector1 - feature_vector2, ord=r) for r in r_values]

plt.plot(r_values, minkowski_distances, marker='o')
plt.xlabel('r values')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs r')
plt.show() #ploting the graph of Minkowski distance

# A4: operation to Divide the dataset into train & test sets
X = df.drop('Category', axis=1)
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# adding categorical labels to numeric labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Scaling the features for avoiding numerical instabilities
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# A5: Training the  kNN classifier (k = 3) with the encoded labels 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_scaled, y_train_encoded)

# A6: finding Test accuracy
accuracy = neigh.score(X_test_scaled, label_encoder.transform(y_test))

# A7: Predict the accuracy using test set
predictions = neigh.predict(X_test_scaled)

# A8: Comparing the kNN (k = 3) with NN (k = 1) by varying k
k_values = range(1, 12)
accuracy_scores = []

for k in k_values:
    neigh_k = KNeighborsClassifier(n_neighbors=k)
    neigh_k.fit(X_train_scaled, y_train_encoded)
    accuracy_k = neigh_k.score(X_test_scaled, label_encoder.transform(y_test))
    accuracy_scores.append(accuracy_k)

plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('k values')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k for kNN')
plt.show() #ploting the graph of comparision

# A9: finding the  confusion matrix and other performance metrics
conf_matrix = confusion_matrix(label_encoder.transform(y_test), predictions)
precision = precision_score(label_encoder.transform(y_test), predictions, average='weighted')
recall = recall_score(label_encoder.transform(y_test), predictions, average='weighted')
f1 = f1_score(label_encoder.transform(y_test), predictions, average='weighted')

print('\nConfusion Matrix:')
print(conf_matrix)

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
