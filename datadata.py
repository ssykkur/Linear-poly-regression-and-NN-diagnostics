import numpy as np

def load_data():
    data = np.loadtxt('c:/Users/ashem/studies/andrewngcourse/c2/w3/labs/data/data_w3_ex1.csv', delimiter=',')
    return data

data = load_data()

