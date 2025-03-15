import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('heart.csv')

# Drop rows with missing values
df = df.dropna()

isMale = df['male'].to_numpy()
age = df['age'].to_numpy()
education = df['education'].to_numpy()
currentSmoker = df['currentSmoker'].to_numpy()
cigsPerDay = df['cigsPerDay'].to_numpy()
BPMeds = df['BPMeds'].to_numpy()
prevalentStroke = df['prevalentStroke'].to_numpy()
prevalentHyp = df['prevalentHyp'].to_numpy()
diabetes = df['diabetes'].to_numpy()
totChol = df['totChol'].to_numpy()
sysBP = df['sysBP'].to_numpy()
diaBP = df['diaBP'].to_numpy()
bmi = df['BMI'].to_numpy()
heartRate = df['heartRate'].to_numpy()
glucose = df['glucose'].to_numpy()

outcome = df['TenYearCHD'].to_numpy()

X = np.column_stack((isMale, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, bmi, heartRate, glucose))
Y = outcome

# Split data into training and testing sets
def test_train_split(X:list, Y:list, splitting_ratio:float) -> tuple:
    X_train = X[:int(splitting_ratio * X.shape[0])]
    Y_train = Y[:int(splitting_ratio * Y.shape[0])]
    X_test = X[int(splitting_ratio * X.shape[0]):]
    Y_test = Y[int(splitting_ratio * Y.shape[0]):]
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = test_train_split(X, Y, 0.8)

# add polynomial features
def poly(X:list, degree:int) -> list:
    X_poly = np.ones((X.shape[0], 1))
    for i in range(1, degree+1):
        X_poly = np.column_stack((X_poly, X**i))
    return X_poly

# normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_poly = poly(X_train_scaled, 2)
X_test_poly = poly(X_test_scaled, 2)

# print(X_train_poly, X_train_poly.shape)

# model
# number of neurons = 29
model = Sequential(
    [
        Input(shape=(31,), name='Input'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid', name='Output')
    ]
)


model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

model.fit(X_train_poly, Y_train, epochs=50)

# evaluate model
predictions = model.predict(X_test_poly)
predictions = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(predictions == Y_test)

print(f"\n\n\nModel weights: {model.get_weights()}\n\n")
print(f"Accuracy: {accuracy}")