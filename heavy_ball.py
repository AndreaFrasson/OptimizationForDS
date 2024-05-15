# implementation of the heavy ball approach
import numpy as np

# norm of the gradient (vector)
# x, gradient
# return n, the norm
def norm(x):
   n = 0
   for i in x:
      n+=np.linalg.norm(i)
   
   return n

# function to update the weights and the biases in our neural network
# model, neural network
# stepsize
# d_i, direction, gradient or a combination
def update(model, stepsize, d_i):
   model.weights_input_hidden += stepsize*d_i[0]
   model.bias_hidden += stepsize*d_i[1]
   model.weights_hidden_output += stepsize*d_i[2]
   model.bias_output += stepsize*d_i[3]


# implementation of the Heavy Ball Approach (Classical momentum), applied to the neural network
# - model, neural network already initialized
# - X, set of predictor attributes, a real matrix n x 10
# - y, set of predicted attribute, a integer matrix n x 12, each vector
#   is a vector representing the label assigned
# - alpha, learning rate, default 5e-2
# - mu, momentum term, default 0.9
# - eps, precision required to stop, default 1e-10
# - MaxIt, maximum number of iteration, default 1000
# - outlist, parameter to select what is to return. If set to True
#       the algorithm returns the list of all the points discovered and the
#       list of all the best points. If set to false, return only the best point
def Heavy_Ball_Approach(model, X, y, alpha=0.05, mu=0.05, MaxIt=1000, outlist = True):
    # initialization
    pg = [0.0]*4
    xref = [model.weights_input_hidden.copy(), model.bias_hidden.copy(), 
            model.weights_hidden_output.copy(), model.bias_output.copy()]
    i = 0
    f_list = []
    f_best =[] 

    # main loop 
    while True:

        #compute function and subgradient
        f_i = model.evaluate(X, y)

        if outlist:
            f_list.append(f_i)
            if len(f_best) > 0:
                f_best.append(min(f_best[-1], f_i))
            else:
                f_best.append(f_i)

        g_i = model.gradient(X, y)
        ng = norm(g_i)

        #exit conditions:
        # max iteration
        if i >= MaxIt:
            break
        
        # the difference between iterations is not significant anymore
        if len(f_list) > 1:
            if np.abs(f_list[-2] - f_list[-1]) < 1e-10:
                xref = [model.weights_input_hidden, model.bias_hidden, model.weights_hidden_output, model.bias_output]
                break
        
        # norm of the gradient smaller than some threshold
        if  ng <= 1e-12:
            xref = [model.weights_input_hidden, model.bias_hidden, model.weights_hidden_output, model.bias_output]
            break
        
        #update:
        #compute direction
        d_i = []
        for j in range(len(g_i)):
            d_i.append(-alpha*g_i[j] + mu*pg[j])
            
        pg = d_i

        # Update weights and biases taking into consideration past direction
        update(model, 1, d_i)

        i = i + 1

    if outlist:
        #return all the points in the descent and all the best points
        return f_list, f_best

    # return point corresponding to best value found so far
    return xref.copy()
