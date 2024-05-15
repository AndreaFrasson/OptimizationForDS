import Neural_Network as nn

import heavy_ball as hb
import defl_sub as ds

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product
import matplotlib.pyplot as plt


# opend the Avila dataset and perform basic cleaning and preparation
def open_data():
    # open dataset
    df = pd.read_table('data/avila-tr.txt', header=None, sep=',')
    # str type for class column
    df[10] = df[10].astype(str)

    # split data into train and test
    X = df.drop(10, axis=1).values
    y = df[10].values

    # one hot encoding of y
    enc = OneHotEncoder(handle_unknown='ignore')
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()

    # min max scaling X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.array(X)

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# grid search procedure for the heavy ball algorithm applied to our neural network
def grid_search_for_heavy_ball(model, X_train, y_train, fix_w1, fix_b1, fix_w2, fix_b2):
    alphas=[0.05, 0.01, 0.005, 0.001, 5e-4, 1e-4, 5e-5, 1e-5] # learning rate
    mus=[0.999, 0.995, 0.9, 0.5, 0.3, 0.1] # mu_max
    rho = 0.001
    hb_res = {}

    cartesian_product = list(product(alphas, mus))
    for alpha, mu in cartesian_product:

        model.weights_input_hidden = fix_w1.copy()
        model.bias_hidden = fix_b1.copy()
        model.weights_hidden_output = fix_w2.copy()
        model.bias_output = fix_b2.copy()

        f,b = hb.Heavy_Ball_Approach(model, X_train, y_train, alpha, mu, MaxIt=1000)
        # final value
        fv = f[-1]
        # iteration
        it = len(f)
        # difference
        diff = np.abs(f[0] - f[-1])
        # convergence speed
        cs = np.where(f <= fv+rho)[0][0]

        hb_res[(alpha, mu)] = [fv, it, diff, cs] 

    # sort by loss values
    hb_res = sorted(hb_res.items(), key = lambda x:x[1][0])

    print(hb_res[0])
    #get parameters for the best configuration
    alpha, mu = hb_res[0][0] # (0.01, 0.9)
    return alpha, mu


# print useful information at the end of the descent
def show_info(descent, rho):
    # final value
    fv = descent[-1]
    # difference
    diff = ((descent[0] - descent[-1]))
    # convergence speed
    cs = np.where(descent <= fv+rho)[0][0]

    print('final value: ', fv)
    print('difference: ', diff)
    print('convergence speed: ', cs)



# random search procedure for the deflected subgradient algorithm applied to our neural network
def random_search_for_deflected(model, X_train, y_train, fix_w1, fix_b1, fix_w2, fix_b2):
    alphas=[0.3, 0.5, 0.7, 0.9, 1] # deflection parameter
    betas = [0.1, 0.5, 0.7, 0.9, 1.5, 1.7, 2] # beta for the stepsize
    astarts = [1e-4, 0.01, 0.1, 0.2] #displacement
    taus = [0.9, 0.95]
    epss = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    defl_res = {}
    rho = 0.001

    i = 0

    while i < 100:

        alpha = np.random.choice(alphas) 
        beta = np.random.choice(betas) 
        astart = np.random.choice(astarts) 
        tau = np.random.choice(taus)
        eps = np.random.choice(epss)

        model.weights_input_hidden = fix_w1.copy()
        model.bias_hidden = fix_b1.copy()
        model.weights_hidden_output = fix_w2.copy()
        model.bias_output = fix_b2.copy()

        f,b = ds.Deflected_Subgradient_Approach(model, X_train, y_train, alpha, beta, eps, astart = astart, tau = tau, MaxIt=1000)
        # final value
        fv = b[-1]
        # iteration
        it = len(f)
        # difference
        diff = np.abs(f[0] - f[-1])
        # convergence speed
        cs = np.where(f <= fv+rho)[0][0]

        defl_res[(alpha, beta, astart, tau, eps)] = [fv, it, diff, cs] 
        i += 1

    # sort by loss values
    defl_res = sorted(defl_res.items(), key = lambda x:x[1][0])

    print(defl_res[0])
    #get parameters for the best configuration
    alpha, beta, astart, tau, eps = defl_res[0][0] 
    return alpha, beta, astart, tau, eps 



def experiments(l1):
    X_train, X_test, y_train, y_test = open_data()

    if l1 == 1e-2:
        fix_w1 = np.load('fix_w1_l1.npy')
        fix_b1 = np.load('fix_b1_l1.npy')
        fix_w2 = np.load('fix_w2_l1.npy')
        fix_b2 = np.load('fix_b2_l1.npy')

        min_loc_hb_l1 = 0.03207727750612194
        min_loc_ds_l1 = 0.032645352600954673

        iterations = 1000
        
    elif l1 == 1:
        fix_w1 = np.load('fix_w1_l2.npy')
        fix_b1 = np.load('fix_b1_l2.npy')
        fix_w2 = np.load('fix_w2_l2.npy')
        fix_b2 = np.load('fix_b2_l2.npy')

        min_loc_hb_l1 = 0.03900939614626004 
        min_loc_ds_l1 = 0.03829861653866613

        iterations = 5000
    
    else:
        print('no experiment made, no local minima searched')
        return

    input_size = X_train.shape[1]
    hidden_size = 4
    output_size = 12
    rho = 0.001

    model = nn.NeuralNetwork(input_size, hidden_size, output_size, l1)
    #alpha, mu = grid_search_for_heavy_ball(model)
    if l1 == 1e-2:
        alpha, mu = (0.01, 0.9)
    else:
        alpha, mu = (0.0001, 0.9)

    model.weights_input_hidden = fix_w1.copy()
    model.bias_hidden = fix_b1.copy()
    model.weights_hidden_output = fix_w2.copy()
    model.bias_output = fix_b2.copy()

    f_hb_l1, b_hb_l1 = hb.Heavy_Ball_Approach(model, X_train, y_train, alpha, mu, MaxIt=iterations)

    print('Heavy Ball results:')
    show_info(b_hb_l1, rho)
    print('')
    d_hb_l1 = []
    for i in b_hb_l1:
        d_hb_l1.append(np.abs(i - min_loc_hb_l1)/min_loc_hb_l1)


    #alpha, beta, eps, astart, tau = random_search_for_deflected(model)
    if l1 == 1e-2:
        alpha, beta, astart, tau, eps = (1.0, 0.1, 0.01, 0.9, 0.1)
    else:
        alpha, beta, astart, tau, eps = (0.5, 1.7, 0.2, 0.95, 1e-05)


    model.weights_input_hidden = fix_w1.copy()
    model.bias_hidden = fix_b1.copy()
    model.weights_hidden_output = fix_w2.copy()
    model.bias_output = fix_b2.copy()
    f_ds_l1, b_ds_l1 = ds.Deflected_Subgradient_Approach(model, X_train, y_train,
                                                alpha, beta, eps, astart, tau, MaxIt = iterations)
    
    print('Deflected Subgradient results:')
    show_info(b_ds_l1, rho)
    print('')

    d_ds_l1 = []
    for i in b_ds_l1:
        d_ds_l1.append(np.abs(i - min_loc_ds_l1)/(min_loc_ds_l1))


    # all plot in the same picture
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
    # generates a converging sequence for heavy ball
    n = iterations
    eps3 = []
    for k in range(1,n):
        eps3.append(1.0/(k))

    axs[0].set_title('Heavy Ball')
    axs[0].semilogy(eps3)
    axs[0].semilogy(d_hb_l1, color = 'red', linestyle = 'dashed')
    axs[0].set_xlabel('Number of iterations')
    axs[0].set_ylabel('Relative Gap')
    axs[0].legend(['Theoretical', 'Empirical'])

    # generates a converging sequence for deflected subgradient
    eps3 = []
    for k in range(1,n):
        eps3.append(1.0/np.sqrt(k))

    axs[1].set_title('Deflected Subgradient')
    axs[1].semilogy(eps3)
    axs[1].semilogy(d_ds_l1, color = 'red', linestyle = 'dashed')
    axs[1].set_xlabel('Number of iterations')
    axs[1].set_ylabel('Relative Gap')
    axs[1].legend(['Theoretical', 'Empirical'])

    axs[2].set_title('Comparison between algorithms')
    axs[2].semilogy(d_ds_l1, color = 'blue', linestyle = 'dashed')
    axs[2].semilogy(d_hb_l1, color = 'red', linestyle = 'dashed', alpha = 0.8)
    axs[2].set_xlabel('Number of iterations')
    axs[2].set_ylabel('Relative Gap')
    axs[2].legend(['Deflected Subgradient', 'Heavy Ball'])
    # Adjust layout
    plt.tight_layout()
    # Show plots
    plt.show()


if __name__ == '__main__':
    experiments(1e-2)