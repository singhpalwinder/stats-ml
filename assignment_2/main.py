import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('breast_cancer_data.csv')

print("First 5 rows:")
print(df.head())
print("\nDataset Summary:")
print(df.describe())


if df.isnull().sum().sum() == 0:
    print("\nNo missing values found.")
else:

    df = df.dropna()


X = df.drop(['ID', 'Diagnosis'], axis=1) 
y = df['Diagnosis'] 


y = y.map({'M': 1, 'B': 0}) if y.dtype == 'O' else y


scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

print("\nData Scaled successfully.")


features_to_plot = ['radius1', 'texture1', 'perimeter1', 'area1']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel() 

for i, feature in enumerate(features_to_plot):
    # Plot Malignant (y==1)
    axes[i].hist(X_scaled[y == 1][feature], color='red', alpha=0.5, label='Malignant', bins=20)
    # Plot Benign (y==0)
    axes[i].hist(X_scaled[y == 0][feature], color='blue', alpha=0.5, label='Benign', bins=20)
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].legend()

plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred_class = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1] 


def my_accuracy(y_true, y_pred):
    
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

def my_binary_cross_entropy(y_true, y_prob):
 
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    
    bce = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return bce

def my_log_likelihood(y_true, y_prob):

    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    
    ll = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return ll

acc = my_accuracy(y_test.values, y_pred_class)
bce = my_binary_cross_entropy(y_test.values, y_pred_prob)
ll = my_log_likelihood(y_test.values, y_pred_prob)

print(f"\n--- Custom Metrics Results ---")
print(f"Accuracy: {acc:.4f}")
print(f"Binary Cross Entropy: {bce:.4f}")
print(f"Log Likelihood: {ll:.4f}")