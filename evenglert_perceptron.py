# # Implementing a Perceptron
# 
# In this task, you will implement a simple perceptron model from scratch using Python. The goal of this task is to give you a hands-on experience of how a perceptron works and how it can be used for binary classification.
# 
# Instructions:
# 
# 1. Define a function called `perceptron` that takes in three parameters:
#     - `X`: a numpy array of shape `(n_samples, n_features)` representing the input data.
#     - `y`: a numpy array of shape `(n_samples,)` representing the target labels (0 or 1).
#     - `eta`: the learning rate for updating the weights.
# 
# 2. Initialize the weight vector `w` to a random value of shape `(n_features,)`.
# 
# 3. Define a for loop that iterates over a specified number of epochs (e.g., 100). Within each epoch, iterate over all the training samples and do the following:
#     - Compute the predicted output `y_pred` by multiplying the input vector `X` with the weight vector `w` and passing it through a step function (e.g., Heaviside function).
#     - Compute the error `err` as the difference between the true labels `y` and predicted output `y_pred`.
#     - Update the weight vector `w` using the following formula: `w = w + eta * err * X[i]`
# 
# 4. After all epochs have been completed, return the learned weight vector `w`.
# 
# 5. Test your perceptron implementation on a simple binary classification problem, such as the XOR problem. Generate random data points and labels for the XOR problem and train your perceptron on this data. Print the learned weights and test the perceptron on new data points to see how well it can classify.
# 
# Bonus:
# 
# - Modify the perceptron to implement the perceptron learning algorithm with a bias term.
# - Modify the perceptron to implement the adaptive linear neuron (Adaline) algorithm.
# 
# Deliverables:
# 
# - A Jupyter notebook or Python script that implements the perceptron from scratch and solves the XOR problem.
# - A brief report explaining your implementation and results.
# 
# Author: Evgeniya Englert
# 
# Last update: 2026-03-23

# --- BEGIN ---

# ===========================================================
# PERCEPTRON IMPLEMENTATION FROM SCRATCH (WITH FULL COMMENTS)
# ===========================================================

# This script implements:
# 1. A basic perceptron for binary classification
# 2. Training on XOR dataset
# 3. Testing predictions
# 4. BONUS:
#    - Perceptron with bias
#    - Adaline (Adaptive Linear Neuron)
# 
# IMPORTANT NOTE:
# The XOR problem is NOT linearly separable, meaning a simple perceptron
# CANNOT perfectly learn it. This is intentional for demonstration.

# Instructions 1-4: Define perceptron
# 1. Define a function called `perceptron` that takes in three parameters:
#     - `X`: a numpy array of shape `(n_samples, n_features)` representing the input data.
#     - `y`: a numpy array of shape `(n_samples,)` representing the target labels (0 or 1).
#     - `eta`: the learning rate for updating the weights.
# 
# 2. Initialize the weight vector `w` to a random value of shape `(n_features,)`.
# 
# 3. Define a for loop that iterates over a specified number of epochs (e.g., 100). Within each epoch, iterate over all the training samples and do the following:
#     - Compute the predicted output `y_pred` by multiplying the input vector `X` with the weight vector `w` and passing it through a step function (e.g., Heaviside function).
#     - Compute the error `err` as the difference between the true labels `y` and predicted output `y_pred`.
#     - Update the weight vector `w` using the following formula: `w = w + eta * err * X[i]`
# 
# 4. After all epochs have been completed, return the learned weight vector `w`.

# DEFINE HELPING FUNCTIONS

# Import NumPy for numerical operations
import numpy as np

# ----------------------------------------------------------
# STEP FUNCTION (Activation Function)
# ----------------------------------------------------------
def step_function(x):
    """
    Heaviside step function.

    This function converts continuous values into binary outputs:
    - Returns 1 if input >= 0
    - Returns -1 if input < 0

    Why -1 and 1 instead of 0 and 1?
    Because perceptron learning rule is cleaner with {-1, +1}.
    """
    return np.where(x >= 0, 1, -1)

# ----------------------------------------------------------
# TESTING FUNCTION
# ----------------------------------------------------------
def predict(X, w):
    """
    Predict class labels for given input.

    PARAMETERS:
    -----------
    X : input data
    w : trained weights

    RETURNS:
    --------
    predictions : array of predicted labels
    """
    # Compute linear combination
    linear_output = np.dot(X, w)

    # Apply step function
    return step_function(linear_output)

# ----------------------------------------------------------
# BASIC PERCEPTRON IMPLEMENTATION
# ----------------------------------------------------------
def perceptron(X, y, eta=0.01, epochs=100):
    """
    Implements a simple perceptron algorithm.

    PARAMETERS:
    -----------
    X : numpy array of shape (n_samples, n_features)
        Input data (features)

    y : numpy array of shape (n_samples,)
        Target labels (-1 or 1)

    eta : float
        Learning rate (controls how big updates are)

    epochs : int
        Number of full passes through the dataset

    RETURNS:
    --------
    w : numpy array
        Learned weight vector
    """

    # ------------------------------------------------------
    # STEP 1: INITIALIZE WEIGHTS RANDOMLY
    # ------------------------------------------------------
    # Create a random weight vector with one weight per feature
    # Small random values help break symmetry
    w = np.random.randn(X.shape[1])

    # Print initial weights for debugging
    print("Initial weights:", w)

    # ------------------------------------------------------
    # STEP 2: TRAINING LOOP (EPOCHS)
    # ------------------------------------------------------
    # Repeat training multiple times over dataset
    for epoch in range(epochs):

        # Loop through each training example
        for i in range(X.shape[0]):

            # --------------------------------------------------
            # STEP 3: COMPUTE LINEAR OUTPUT
            # --------------------------------------------------
            # Dot product between input vector and weights
            # This represents the "activation" before threshold
            linear_output = np.dot(X[i], w)

            # --------------------------------------------------
            # STEP 4: APPLY STEP FUNCTION
            # --------------------------------------------------
            # Convert continuous value to binary class
            y_pred = step_function(linear_output)

            # --------------------------------------------------
            # STEP 5: COMPUTE ERROR
            # --------------------------------------------------
            # Difference between true label and predicted label
            err = y[i] - y_pred

            # --------------------------------------------------
            # STEP 6: UPDATE WEIGHTS
            # --------------------------------------------------
            # Perceptron update rule:
            # w = w + eta * error * input_vector
            #
            # Intuition:
            # - If prediction is correct → err = 0 → no update
            # - If wrong → adjust weights toward correct direction
            w = w + eta * err * X[i]

        # Optional: print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch} completed")

    # ------------------------------------------------------
    # STEP 7: RETURN FINAL WEIGHTS
    # ------------------------------------------------------
    return w

# Instruction 5. Test your perceptron implementation on a simple binary classification problem, such as the XOR problem. Generate random data points and labels for the XOR problem and train your perceptron on this data. Print the learned weights and test the perceptron on new data points to see how well it can classify.

# ----------------------------------------------------------
# GENERATE XOR DATASET
# ----------------------------------------------------------
np.random.seed(0)  # For reproducibility

# Generate 100 random 2D points
X = np.random.randn(100, 2)

# XOR condition:
# True when one coordinate is positive and the other is negative
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Convert True/False → 1 / -1
y = np.where(y, 1, -1)

# ----------------------------------------------------------
# TRAIN THE PERCEPTRON
# ----------------------------------------------------------
weights = perceptron(X, y, eta=0.01, epochs=100)

print("\nLearned weights (no bias):", weights)

# Test on training data
predictions = predict(X, weights)

# Calculate accuracy
accuracy = np.mean(predictions == y)

print("Training accuracy (no bias):", accuracy)

# Bonus:
# 
# - Modify the perceptron to implement the perceptron learning algorithm with a bias term.
# - Modify the perceptron to implement the adaptive linear neuron (Adaline) algorithm.

# ==========================================================
# BONUS 1: PERCEPTRON WITH BIAS (FULL PIPELINE)
# ==========================================================
def perceptron_with_bias(X, y, eta=0.01, epochs=100):
    """
    Perceptron with bias term.

    The bias allows the decision boundary to shift away from origin,
    making the model more flexible.
    """

    # ------------------------------------------------------
    # INITIALIZE WEIGHTS AND BIAS
    # ------------------------------------------------------
    w = np.random.randn(X.shape[1])  # weight vector
    b = np.random.randn()            # scalar bias

    print("\nInitial weights (with bias):", w)
    print("Initial bias:", b)

    # ------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------
    for epoch in range(epochs):

        for i in range(X.shape[0]):

            # Compute linear combination INCLUDING bias
            linear_output = np.dot(X[i], w) + b

            # Apply step function
            y_pred = step_function(linear_output)

            # Compute error
            err = y[i] - y_pred

            # Update weights
            w = w + eta * err * X[i]

            # Update bias (IMPORTANT difference!)
            b = b + eta * err

        # Optional progress print
        if epoch % 10 == 0:
            print(f"[Bias Perceptron] Epoch {epoch} completed")

    return w, b

# ----------------------------------------------------------
# TRAIN PERCEPTRON WITH BIAS
# ----------------------------------------------------------
w_b, b_b = perceptron_with_bias(X, y, eta=0.01, epochs=100)

print("\nLearned weights (with bias):", w_b)
print("Learned bias:", b_b)

# ----------------------------------------------------------
# PREDICTION FUNCTION (WITH BIAS)
# ----------------------------------------------------------
def predict_with_bias(X, w, b):
    """
    Predict labels using trained perceptron with bias.

    PARAMETERS:
    -----------
    X : input data
    w : learned weights
    b : learned bias

    RETURNS:
    --------
    predictions : predicted class labels
    """

    # Compute linear output for all samples
    linear_output = np.dot(X, w) + b

    # Apply step function
    return step_function(linear_output)

# ----------------------------------------------------------
# TESTING THE MODEL
# ----------------------------------------------------------
predictions_bias = predict_with_bias(X, w_b, b_b)

# ----------------------------------------------------------
# CALCULATE ACCURACY
# ----------------------------------------------------------
accuracy_bias = np.mean(predictions_bias == y)

print("\nTraining accuracy (with bias):", accuracy_bias)

# ----------------------------------------------------------
# TEST ON NEW DATA POINTS
# ----------------------------------------------------------
# Define some new test points manually
X_new = np.array([
    [1, 1],
    [-1, -1],
    [1, -1],
    [-1, 1]
])

# Get predictions
new_preds = predict_with_bias(X_new, w_b, b_b)

print("\nNew test samples:")
print(X_new)

print("Predictions for new samples:")
print(new_preds)

# ==========================================================
# BONUS 2: ADALINE (Adaptive Linear Neuron)
# ==========================================================
def adaline(X, y, eta=0.01, epochs=100):
    """
    Adaline algorithm.

    Difference from perceptron:
    - Uses linear output directly (NO step function during training)
    - Minimizes Mean Squared Error (MSE)
    """

    # Initialize weights and bias
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

    for epoch in range(epochs):

        # Compute linear outputs for ALL samples
        linear_output = np.dot(X, w) + b

        # Compute errors (continuous!)
        errors = y - linear_output

        # Update weights using gradient descent
        w += eta * np.dot(X.T, errors)

        # Update bias
        b += eta * np.sum(errors)

    return w, b

# Train Adaline
w_a, b_a = adaline(X, y)

print("\nAdaline weights:", w_a)
print("Adaline bias:", b_a)

# ----------------------------------------------------------
# TEST ADALINE
# ----------------------------------------------------------
def predict_adaline(X, w, b):
    """
    Prediction for Adaline:
    Apply step function AFTER training
    """
    linear_output = np.dot(X, w) + b
    return step_function(linear_output)


preds_adaline = predict_adaline(X, w_a, b_a)
acc_adaline = np.mean(preds_adaline == y)

print("Adaline accuracy:", acc_adaline)

# --- END ---


