# import modules and libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("P:\LP\DL Datasets\BostonHousingDataset.csv")

# Display the first few rows of the dataset
print(df.head())

# Step 2: Preprocess the data
X = df.drop('medv', axis=1) #axis 1 specifies dropping the col
y = df['medv']

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the first few rows of the scaled input features
print(X_scaled[:5])

# Step 3: Split the dataset
# test_size parameter specifies the proportion of the dataset that should be included in the testing set
# By setting a fixed value for random_state, you ensure that:
# The same samples are assigned to the training and testing sets each time the code is run.
# The order of the samples in the dataset does not affect the splitting outcome.
# X_scaled are features and y is target variable
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Print the shapes of the training and testing sets
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Step 4: Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Display the model summary
print(model.summary())

# Step 5: Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# Step 6: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()

# Step 7: Evaluate the model
predictions = model.predict(X_test)
loss, mae = model.evaluate(X_test, y_test)

# Reshape predictions to match the shape of y_test
predictions = predictions.flatten()

# Calculate RMSE
import numpy as np
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

# Print both MAE and RMSE
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
