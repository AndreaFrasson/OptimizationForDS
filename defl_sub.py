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
   model.weights_input_hidden -= stepsize*d_i[0]
   model.bias_hidden -= stepsize*d_i[1]
   model.weights_hidden_output -= stepsize*d_i[2]
   model.bias_output -= stepsize*d_i[3]
   

# implementation of the Deflected Subgradient apporach, applied to a function
# - model, neural network already initialized
# - X, set of predictor attributes, a real matrix n x 10
# - y, set of predicted attribute, a integer matrix n x 12, each vector
# - alpha, deflection parameter
# - beta, stepsize scale factor
# - eps, the minimum relative value for the displacement 
# - astart, relative value to which the displacement is reset each time
# - tau, a target-level Polyak stepsize with nonvanishing threshold is used,
#       then delta^{i + 1} = delta^i * tau each time
#       f( x^{i + 1} ) > f^i_{ref} - delta^i
# - MaxIt, maximum number of iteration, default 1000
# - outlist, parameter to select what is to return. If set to True
#       the algorithm returns the list of all the points discovered and the
#       list of all the best points. If set to false, return only the best point
def Deflected_Subgradient_Approach(model, X, y, alpha = 0.7, beta = 1, eps =1e-6, 
                                   astart = 1e-4 , tau = 0.95, MaxIt= 1000, outlist = True):
    #initialization
    pg = [0.0]*4 # initialize vector of past directions

    f_list = [] # every value computed
    f_best = [] # only best values
    i = 1
    #starting point
    xref = [model.weights_input_hidden.copy(), model.bias_hidden.copy(), 
            model.weights_hidden_output.copy(), model.bias_output.copy()]
    fref = float('inf')   #best f-value found so far
    delta = 0;  #required displacement from fref;
    
    #main loop 
    while True:

        #compute function and subgradient
        f_i = model.evaluate(X, y)

        #append the right values in the output lists
        if outlist:
            f_list.append(f_i)
            if len(f_best) > 0:
                f_best.append(min(f_best[-1], f_i))
            else:
                f_best.append(f_i)

        g_i = model.gradient(X, y)
        ng = norm(g_i)

        # stopping criteria 
        if ng < 1e-12: # optimal
            xref = [model.weights_input_hidden.copy(), model.bias_hidden.copy(), 
                    model.weights_hidden_output.copy(), model.bias_output.copy()]
            break
        
        if i > MaxIt: # stopped
            break
        
        # update
        if f_i <= fref - delta:  #found a "significantly" better point
            delta = astart * max(abs( f_i ) , 1);  #reset delta
        else: #decrease delta
            delta = max(delta * tau ,  eps * max(abs(min(f_i, fref)) , 1))
                
        if f_i < fref:    #found a better f-value 
            fref = f_i  #update fref
            xref = [model.weights_input_hidden.copy(), model.bias_hidden.copy(), 
                    model.weights_hidden_output.copy(), model.bias_output.copy()]  #this is the current solution

        # compute deflection
        d_i = []
        for j in range(len(g_i)):
            d_i.append(alpha*g_i[j] + (1-alpha)*pg[j])
            
        pg = d_i #store 'new' past directions
        nd = norm(d_i) #compute the norm for the stepsize

        # compute stepsize
        # Polyak stepsize with target level
        a = beta * ( f_i - fref + delta ) / ( nd * nd )

        # compute new point
        update(model, a, d_i)

        i = i + 1

    if outlist:
        #return all the points in the descent and all the best points
        return f_list, f_best

    # return point corresponding to best value found so far
    return xref.copy()




