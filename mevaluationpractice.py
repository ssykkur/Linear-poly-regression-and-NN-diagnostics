import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datadata import data
import tensorflow as tf
import time
import utils

np.set_printoptions(precision=2)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# For linear models
data = np.loadtxt('c:/Users/ashem/studies/andrewngcourse/c2/w3/labs/data/data_w3_ex1.csv', delimiter=',')
x = data[:,0]
y = data[:,1]
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# For binary classification models
data_2 = np.loadtxt('c:/Users/ashem/studies/andrewngcourse/c2/w3/labs/data/data_w3_ex2.csv', delimiter=',')


x_bc = data_2[:,:-1]
y_bc = data_2[:,-1]
y_bc = np.expand_dims(y_bc, axis=1)
print(f"the shape of the inputs x is: {x_bc.shape}")
print(f"the shape of the targets y is: {y_bc.shape}")


#utils.plot_dataset(x=x, y=y, title='input vs target')
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

del x_, y_

def linear_model(x_train, y_train, x_cv, y_cv, x_test, y_test):
    
    scaler_linear = StandardScaler()
    linear_model = LinearRegression()

    x_train_scaled = scaler_linear.fit_transform(x_train)
    #utils.plot_dataset(x=x_train_scaled, y=y_train, title="scaled input vs. target")
    linear_model.fit(x_train_scaled, y_train)
    yhat_train = linear_model.predict(x_train_scaled)
    mse_train = mean_squared_error(y_train, yhat_train) / 2


    print(f"training MSE (using sklearn function): {mse_train}")
    
    x_cv_scaled = scaler_linear.transform(x_cv)
    yhat_cv = linear_model.predict(x_cv_scaled)

    mse_cv = mean_squared_error(y_cv, yhat_cv) / 2
    print(f"Cross validation MSE: {mse_cv}")

    return mse_train, mse_cv


def poly_model(x_train, y_train, x_cv, y_cv, x_test, y_test):

    poly = PolynomialFeatures(degree=2, include_bias=False)
    model = LinearRegression()
    scaler_poly = StandardScaler()

    X_train_mapped = poly.fit_transform(x_train)
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

    model.fit(X_train_mapped_scaled, y_train)

    yhat_train = model.predict(X_train_mapped_scaled)
    mse_train = mean_squared_error(y_train, yhat) / 2
    print(f"Training MSE: {mse_train}")

    x_cv_mapped = poly.transform(x_cv)
    x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
    yhat_cv = model.predict(x_cv_mapped_scaled)

    mse_cv = mean_squared_error(y_cv, yhat_cv)/2
    print(f'Cv MSE: {cv_mse}')

    return mse_train, mse_cv

#mse_linear_train, mse_linear_cv = linear_model(x_train, y_train, x_cv, y_cv, x_test, y_test)


# Testin for the optimal polynomial degree for features
def multiple_degree_models(x_train, y_train, x_cv, y_cv, x_test, y_test):

    train_mses = []
    cv_mses = []
    models = []
    polys = []
    scalers = []
    for degree in range(1,11):
        
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)
        polys.append(poly)
        
        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_mapped_scaled, y_train )
        models.append(model)
        
        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)
        
        # Add polynomial features and scale the cross validation set
        X_cv_mapped = poly.transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
        
        # Compute the cross validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)


    degree = np.argmin(cv_mses) + 1
    print(f"Lowest CV MSE is found in the model with degree={degree}")
    X_test_mapped = polys[degree-1].transform(x_test)
    X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

    yhat = models[degree-1].predict(X_test_mapped_scaled)
    test_mse = mean_squared_error(y_test, yhat) / 2
    print(f"Training MSE: {train_mses[degree-1]:.2f}")
    print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
    print(f"Test MSE: {test_mse:.2f}")

    return train_mses, cv_mses, models, polys, scalers


def nn_models_linear(x_train, y_train, x_cv, y_cv, x_test, y_test, degree=1):
    # Add polynomial features
    degree = 1
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    X_cv_mapped = poly.transform(x_cv)
    X_test_mapped = poly.transform(x_test)

    # Scale the features using the z-score
    scaler = StandardScaler()
    X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
    X_test_mapped_scaled = scaler.transform(X_test_mapped)
    # Initialize lists that will contain the errors for each model
    nn_train_mses = []
    nn_cv_mses = []

    # Build the models
    nn_models = utils.build_models()

    for model in nn_models:
        
        # Setup the loss and optimizer
        model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        )

        print(f"Training {model.name}...")
        
        # Train the model
        model.fit(
            X_train_mapped_scaled, y_train,
            epochs=300,
            verbose=0
        )
        
        print("Done!\n")

        # Record the training MSEs
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        nn_train_mses.append(train_mse)
        
        # Record the cross validation MSEs 
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        nn_cv_mses.append(cv_mse)


    print("RESULTS:")
    for model_num in range(len(nn_train_mses)):
        print(
            f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
            f"CV MSE: {nn_cv_mses[model_num]:.2f}"
            )

    least_mse = np.argmin(nn_cv_mses)
    model_num = least_mse + 1

    yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
    test_mse = mean_squared_error(y_test, yhat) / 2

    print(f"Selected Model: {model_num}")
    print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
    print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
    print(f"Test MSE: {test_mse:.2f}")

    return nn_train_mses, nn_cv_mses, nn_models, test_mse, model_num


def nn_models_binary(x_bc, y_bc):

    x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

    # Delete temporary variables
    del x_, y_

    scaler_linear = StandardScaler()

    x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
    x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
    x_bc_test_scaled = scaler_linear.transform(x_bc_test)

    nn_train_error = []
    nn_cv_error = []

    models_bc = utils.build_models()

    for model in models_bc:

        model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        )

        print(f"Training {model.name}...")

    # Train the model
        model.fit(
            x_bc_train_scaled, y_bc_train,
            epochs=200,
            verbose=0
        )
        
        print("Done!\n")

        threshold = 0.5
        
        yhat = model.predict(x_bc_train_scaled)
        yhat = tf.math.sigmoid(yhat)
        yhat = np.where(yhat >= threshold, 1, 0)
        train_error = np.mean(yhat != y_bc_train)
        nn_train_error.append(train_error)

        yhat = model.predict(x_bc_cv_scaled)
        yhat = tf.math.sigmoid(yhat)
        yhat = np.where(yhat >= threshold, 1, 0)
        cv_error = np.mean(yhat != y_bc_cv)
        nn_cv_error.append(cv_error)
    
    for model_num in range(len(nn_train_error)):
        print(
            f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
            f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
            )
    
    least_error = np.argmin(nn_cv_error)
    model_num = least_error + 1

    yhat = models_bc[model_num-1].predict(x_bc_test_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    nn_test_error = np.mean(yhat != y_bc_test)

    print(f"Selected Model: {model_num}")
    print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
    print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
    print(f"Test Set Classification Error: {nn_test_error:.4f}")
        

"""
train_mses, cv_mses, models, polys, scalers = multiple_degree_models(x_train, y_train, x_cv, y_cv, x_test, y_test)
degrees=range(1,11)
utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")



nn_train_mses, nn_cv_mses, nn_models, test_mse, model_num = nn_models_linear(x_train, y_train, x_cv, y_cv, x_test, y_test, degree=1)
"""



nn_models_binary(x_bc, y_bc)
