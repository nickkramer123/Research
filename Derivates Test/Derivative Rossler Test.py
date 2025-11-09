# Rossler Attractor
# TODO Taking a long time, shorten tset?
import os
import numpy as np
import pickle
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy import stats

# parameters
a = .2
# ADJUSTABLE to an extent
b_list = np.linspace(.1, 2, 1000) # 250 values between 0 and 2
g = 5.7

# initial variable values
# good default, may need to change later
initVars = [1, 1, 1]

# generate a solution at 3000 evenly spaced samples in the interval 0 <= t <= 250.
tset = np.linspace(0, 250, 3000) # ADJUSTABLE

# FIRST ORDER OF BUISNESS
def rossler(initVars, t, alpha, beta, gamma):
    x = initVars[0]
    y = initVars[1]
    z = initVars[2]
    dxdt = - y - z # CHANGE MADE
    dydt = x + alpha * y
    dzdt = beta + z * (x - gamma) # THERE MAY BE TYPO HERE, CHANGED NOW
    return [dxdt, dydt, dzdt]


# func: the function that computes the derivatives
# y0: array, initial condition of x, y, z
# t: array, sequence of time points to solve for x, y, z
# args: extra arguments to pass through to function (alpha, beta, gamma)

# solutions in a dictionary, a list for each b value
sol_dict = {}

# create empty lists for bifurcation diagram
b_vals_x = []
b_vals_z = []
z_steady_state =[] # long term (post-transient) values of z
x_steady_state =[] # long term (post-transient) values of x

b_vals_dx = []
b_vals_dz = []
dz_steady_state =[]
dx_steady_state =[] 


fsol_dict = {}

for i, b in enumerate(b_list):
    sol = odeint(
        func = rossler,
        y0 = initVars,
        t = tset,
        args = (a, b, g),
        atol = 1e-10,
        rtol = 1e-10
    )
    sol_dict[b] = sol




    # Remove pre-transients
    post_trans_dict = {}
    for b, arr in sol_dict.items():
        half = int(0.5 * len(arr))
        post_trans_dict[b] = arr[half:, :]

    # extract values for graphs
    x_values = post_trans_dict[b][:, 0]
    y_values = post_trans_dict[b][:, 1]
    z_values = post_trans_dict[b][:, 2]


    # Compute derivative time series and extract values for graphs
    derivative_dict = {}
    for i, (b, arr) in enumerate(post_trans_dict.items()):
        dxdt_list = [rossler(row, 0, a, b, g) for row in arr]
        derivative_dict[b] = np.array(dxdt_list)

    # extract values for graphs
    dx_dt = derivative_dict[b][:, 0]
    dy_dt = derivative_dict[b][:, 1]
    dz_dt = derivative_dict[b][:, 2]


    # PLOTS FOR DERIVATIVES
    # 3D Plots
    # For 5 different indexes
    # if i % 50 == 0: 
    #     fig = plt.figure() # create figure
    #     ax = fig.add_subplot(111, projection='3d') # add new axis
    #     ax.scatter(xs = dx_dt, ys = dy_dt, zs = dz_dt, zdir='z', s=20, c=None, depthshade=True)
    #     ax.set_xlabel("X Axis")
    #     ax.set_ylabel("Y Axis")
    #     ax.set_zlabel("Z Axis")
    #     ax.set_title('Attractor at β = ' + f'{b:.3f}' + ':')

    #     plt.show()

    # ADJUSTABLE
    # collect z values
    dzPeaksInd, _ = find_peaks(dz_dt) # find local maximum indices
    dzPeaks = dz_dt[dzPeaksInd] # find values correlated to indices
    dz_steady_state.extend(dzPeaks) # store values
    b_vals_dz.extend([b] * len(dzPeaks))# store corresponding b values

    # collect x values
    dxPeaksInd, _ = find_peaks(dx_dt) # find local maximum indices
    dxPeaks = dx_dt[dxPeaksInd] # find values correlated to indices
    dx_steady_state.extend(dxPeaks) # store values
    b_vals_dx.extend([b] * len(dxPeaks))# store corresponding b values





# # make bifurcation diagram for z
plt.scatter(x = b_vals_dz, y = dz_steady_state, s = .3, alpha=.3, color="black")
plt.xlabel("β (beta)")
plt.ylabel("z (Steady-State Values)")
plt.title("Bifurcation Diagram of the Rössler System for z")
plt.grid(True)
plt.show()

# # make bifurcation diagram for x
plt.scatter(x = b_vals_dx, y = dx_steady_state, s = .3, alpha=.3, color="brown")
plt.xlabel("β (beta)")
plt.ylabel("x (Steady-State Values)")
plt.title("Bifurcation Diagram of the Rössler System for x")
plt.grid(True)
plt.show()



# for nn data collection
print(np.shape(x_values)) # list of 3000 values

# global normalization
# warning: each beta value has its own standardization
dz_sol_dict = {}
for key, arr in derivative_dict.items():
    if np.std(arr) == 0:
        dz_sol_dict[key] = np.zeros_like(arr) # should never happen
    else:
        dz_sol_dict[key] = stats.zscore(arr)


os.makedirs("./Research/data", exist_ok=True)  # creates the folders if missing
with open("./Research/data/valuesDeriv.pkl", "wb") as file:
    pickle.dump(dz_sol_dict, file)







