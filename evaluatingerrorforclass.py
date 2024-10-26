import numpy as np


probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])
predictions = np.where(probabilities >= 0.5, 1, 0)

ground_truth = np.array([1, 1, 1, 1, 1])

misclassified = 0

num_predictions = len(predictions)

for i in range(num_predictions):
    if predictions[i] != ground_truth[i]:
        misclassified += 1 

fractional_error = misclassified/num_predictions

yhat = np.array([0, 1, 1, 0, 1])
average = np.mean(yhat != ground_truth)
print(average)