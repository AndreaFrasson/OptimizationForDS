import numpy as np
import matplotlib.pyplot as plt

def matyas_function(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def matyas_gradient(x):
    grad_x = 0.52 * x[0] - 0.48 * x[1]
    grad_y = 0.52 * x[1] - 0.48 * x[0]
    return np.array([grad_x, grad_y])


def gradient_descent(x_start, y_start, learning_rate, num_iterations):
    x_current = x_start
    y_current = y_start
    path = [(x_current, y_current)]
    
    for i in range(num_iterations):
        x_gradient = 0.52 * x_current - 0.48 * y_current
        y_gradient = 0.52 * y_current - 0.48 * x_current
        
        x_current -= learning_rate * x_gradient
        y_current -= learning_rate * y_gradient
        
        path.append((x_current, y_current))
    
    return path


# implementation of the Heavy Ball approach (Classical momentum), applied to a function
# - f, function
# - x_start, starting point
# - g, gradient of the function
# - alpha, stepsize, default 5e-2
# - mu, momentum term, default 0.9
# - eps, precision required to stop, default 1e-10
# - MaxIt, maximum number of iteration, default 1000
def Heavy_Ball_Approach(f, x, g, alpha=0.05, mu=0.05, eps=1e-10, MaxIt=1000):
   # initialization to 0 of the past directions
   pg = 0

   i = 0
   x_current = x.copy()
   path = [x.copy()]

   # main loop 
   while True:

      f_i = f(x_current)
      #compute gradient
      g_i = g(x_current)
      # norm of the gradient
      ng = np.linalg.norm(g_i)

      i = i+1

      # --------------- exit conditions: ---------------
      # max iteration
      if i >= MaxIt:
         break
      
      # norm of the gradient smaller than some threshold
      if  ng <= eps:
         break
      
      # -------------------- update: ---------------------
      # direction
      d_i = -alpha*g_i + mu*pg
      # store past direction
      pg = d_i.copy()

      # new point 
      x_current += d_i

      path.append(x_current.copy()) # in this version we are interested in all the descent
   
   return path


# implementation of the Deflected Subgradient apporach, applied to a function
# - f, function
# - x, starting point
# - g, gradient of the function
# - alpha, deflection parameter
# - beta, stepsize scale factor
# - eps, the minimum relative value for the displacement, 
# - astart, relative value to which the displacement is reset each time
# - tau, a target-level Polyak stepsize with nonvanishing threshold is used,
#  then delta^{i + 1} = delta^i * tau each time
#     f( x^{i + 1} ) > f^i_{ref} - delta^i
# - MaxIt, maximum number of iteration, default 1000
def Deflected_Subgradient(f , x, g , alpha = 0.7, beta = 1, eps =1e-6 , astart = 1e-4 , tau = 0.95, MaxIt = 1000):
   #initialization
   fStar = -float('inf')
   pg = 0
   path = [x.copy()]

   i = 0
   xref = x
   fref = float('inf')   #best value found so far
   if eps > 0:
      delta = 0
   
   #main loop 
   while True:

      #----------- compute function and subgradient: --------------
      f_i = f(x)
      g_i = g(x)
      # norm of the gradient
      ng = np.linalg.norm(g_i)
    
      # --------------- exit conditions: --------------------------
      if ng < 1e-12:  #unlikely, but it could happen
         xref = x
         break
      
      # max iteration
      if i >= MaxIt:
         break
      # -------------------- update: ------------------------------

      if f_i <= fref - delta:  #found a "significantly" better point
         delta = astart * max( [ abs( f_i ) , 1 ] );  #reset delta
      else: #decrease delta
         delta = max(delta * tau ,  eps * max(abs( min(f_i, fref)) , 1 ))
            
      if f_i < fref:    #found a better f-value 
         fref = f_i  #update fref
         xref = x   #this is the current solution
         path.append(x.copy())


      # deflection
      d_i = alpha*g_i + (1-alpha)*pg
      pg = d_i.copy()
      ng = np.linalg.norm(d_i)

      # stepsize
      # Polyak stepsize with target level
      a = beta * ( f_i - fref + delta ) / ( ng * ng )

      # new point
      x = x - a * g_i

      #iterate 
      i = i + 1
      
   return path



if __name__ == '__main__':
   # Set starting point and parameters
   x_start = -2.4
   y_start = 4.5
   learning_rate = 0.3
   num_iterations = 1500
   print('--------- Gradient Descent ---------')
   gr = gradient_descent(x_start, y_start, learning_rate, num_iterations)
   final_point = gr[-1]
   print("Final point:", final_point)
   print("Value of the Matyas function at the final point:", matyas_function(final_point))
   print('')

   print('--------- Heavy Ball ---------')
   hb = Heavy_Ball_Approach(matyas_function, np.array([x_start, y_start]), matyas_gradient, alpha = 0.3, mu=0.8, MaxIt= num_iterations)
   final_point = hb[-1]
   print("Final point for heavy ball:", final_point)
   print("Value of the Matyas function at the final point:", matyas_function(final_point))
   print('')

   print('--------- Deflected Subgradient ---------')
   ds = Deflected_Subgradient(matyas_function, np.array([x_start, y_start]), matyas_gradient, alpha = 0.9, beta = 1, eps =0.001, astart = 0.1 , tau = 0.99, MaxIt = num_iterations)
   final_point = ds[-1]
   print("Final point deflected subgradient:", final_point)
   print("Value of the Matyas function at the final point:", matyas_function(final_point))


   ### create the plot with less iterations

   # Define range for x and y
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)

   # Create grid of x and y values
   X, Y = np.meshgrid(x, y)

   # Calculate Matyas function values for each point in the grid
   Z = matyas_function([X, Y])

   # Set starting point and parameters for gradient descent
   learning_rate = 0.7
   num_iterations = 75

   fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

   # Plot the contour plot
   axs[0].contour(X, Y, Z, levels=20)  # You can adjust the number of contour levels
   axs[0].set_xlabel('x')
   axs[0].set_ylabel('y')
   axs[0].set_title('Matyas Function Level Set, Gradient Path')
   axs[0].grid(True)

   # Perform gradient descent
   descent_path = gradient_descent(x_start, y_start, learning_rate, num_iterations)
   final_point = descent_path[-1]

   # Extract x and y coordinates from the descent path
   x_gr = [point[0] for point in descent_path]
   y_gr = [point[1] for point in descent_path]

   # Plot the descent path
   axs[0].plot(x_gr, y_gr, color='black', markersize=1, label='Gradient Descent Path')

   # Plot the contour plot
   axs[1].contour(X, Y, Z, levels=20)  # You can adjust the number of contour levels
   axs[1].set_xlabel('x')
   axs[1].set_title('Matyas Function Level Set, Heavy Ball Path')
   axs[1].grid(True)

   # Perform gradient descent
   descent_path = Heavy_Ball_Approach(matyas_function, np.array([x_start, y_start]), matyas_gradient, alpha = 0.3, mu=0.7, MaxIt= num_iterations)
   final_point = descent_path[-1]

   # Extract x and y coordinates from the descent path
   x_gr = [point[0] for point in descent_path]
   y_gr = [point[1] for point in descent_path]

   # Plot the descent path
   axs[1].plot(x_gr, y_gr, color='black', markersize=1, label='Gradient Descent Path')


   # Plot the contour plot
   axs[2].contour(X, Y, Z, levels=20)  # You can adjust the number of contour levels
   axs[2].set_xlabel('x')
   axs[2].yaxis.set_label_position("right")
   axs[2].set_ylabel('y')
   axs[2].set_title('Matyas Function Level Set, Deflected Subgradient Path')
   axs[2].grid(True)

   # Perform gradient descent
   descent_path = Deflected_Subgradient(matyas_function, np.array([x_start, y_start]), matyas_gradient, alpha = 0.8, beta = 0.5, eps =0.1, tau = 0.95, MaxIt = num_iterations)
   final_point = descent_path[-1]

   # Extract x and y coordinates from the descent path
   x_gr = [point[0] for point in descent_path]
   y_gr = [point[1] for point in descent_path]

   # Plot the descent path
   axs[2].plot(x_gr, y_gr, color='black', markersize=1, label='Gradient Descent Path')

   # Adjust layout
   plt.tight_layout()
   # Show plots
   plt.show()
