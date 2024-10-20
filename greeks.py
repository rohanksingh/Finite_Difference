import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import streamlit as st

st.title("Greeks")

# st.write("---")

def f(x):
    return np.exp(x)

def forward_difference(f, x, dx):
    return (f(x+dx)- f(x))/dx


deltasxs= [1.5,1,0.5, 0.2, 0.1]
nums= np.arange(1,5,0.01)

fig1, ax= plt.subplots(figsize=(8,6))


ax.plot(nums, f(nums), label= 'True' , linewidth=3)

for delt in deltasxs:
    ax.plot(nums, forward_difference(f, nums, delt),  label=f"Delta= {delt}", linewidth=2, linestyle= '--')
    

ax.set_ylabel("Derivative")
ax.set_xlabel('$x$')
ax.set_title('Forward Finite Difference')
ax.legend()
# plt.show()
st.pyplot(fig1)

#  calculate a backwards difference, 
#  this is done by taking a finite step backwards on the function and 
#  seeing how much it has changed from the current value 

def backward_difference(f, x,dx):
    return (f(x)- f(x-dx))/dx

nums= np.arange(1,5,0.01)
true=f(nums)
deltaxs= [1.5,1,0.5,0.2,0.1]

fig2, ax= plt.subplots(figsize=(8,6))

ax.plot(nums, true, label='True', linewidth=3)
for delt in deltasxs:
    ax.plot(nums, backward_difference(f, nums, delt),
             label=f"Delta = {delt}", linewidth=2, linestyle='--')
ax.legend()
ax.set_ylabel("Derivative")
ax.set_xlabel('$x$')
ax.set_title('Backward Finite Difference')
# plt.show()
st.pyplot(fig2)

# Central Difference 
# A central difference combines both the forward and backward difference methods explained above and takes an average of the two. 
# This is known as a second order finite difference, it shouldn't be surprising that this will be a more accurate approximation. 

def central_difference(f, x, dx):
    return (f(x+dx)- f(x-dx))/ (2*dx)



deltaxs= [1.5,1,0.5,0.2,0.1]
fig3, ax= plt.subplots(figsize= (8,6))

ax.plot(nums, true, label='True', linewidth=3)
for delt in deltasxs:
    ax.plot(nums, central_difference(f, nums, delt),
             label=f"Delta = {delt}", linewidth=2, linestyle='--')
ax.legend()
ax.set_ylabel("Derivative")
ax.set_xlabel('$x$')
ax.set_title('Central Finite Difference')
# plt.show()
st.pyplot(fig3)



# if deltaxs== deltafd:
#     st.success("central_difference")
    
# col1, col2= st.columns(2, gap='large')
# with col1:
#     st.header('Central Difference')
    

# from blackscholes import BS_CALL, BS_PUT


# # Greeks

# K = 100
# r= 0.1
# T= 1
# sigma= 0.3 

# S= np.arange(60,140,0.1)
# # print(S)

# calls = ([BS_CALL(s, K , T, r, sigma) for s in S])
# puts= [BS_PUT(s, K, T, r, sigma) for s in S]
# plt.plot(S, calls, label='Call Value')
# plt.plot(S, puts, label='put Value')

# plt.xlabel('So')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# from scipy.stats import norm
# import numpy as np

# def d1(S,K,T,r,sigma):
#     return (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))

# def d2(S, K, T, r, sigma):
#     return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

# def delta_call(S,K, T, r, sigma):
#     N= norm.cdf
#     return N(d1(S,K,T,r,sigma))

# def delta_fdm_call(S,K,T,r,sigma, ds= 1e-5, method='central'):
#     method= method.lower()
#     if method== 'central':
#         return (BS_CALL(S+ds, K, T, r, sigma) - BS_CALL(S-ds, K, T, r, sigma))/(2*ds)
    
#     elif method== 'forward':
#         return (BS_CALL(S+ds, K, T, r, sigma)- BS_CALL(S,K,T,r,sigma))/ds
#     elif method== 'backward':
#         return (BS_CALL(S,K,T,r,sigma)- BS_CALL(S-ds, K , T, r, sigma))/ds
    
# def delta_put(S,K, T, r, sigma):
#     N= norm.cdf
#     return -N(-d1(S,K,T,r,sigma))

# def delta_fdm_put(S,K,T,r,sigma, ds= 1e-5, method='central'):
#     method= method.lower()
#     if method== 'central':
#         return (BS_PUT(S+ds, K, T, r, sigma) - BS_PUT(S-ds, K, T, r, sigma))/(2*ds)
    
#     elif method== 'forward':
#         return (BS_PUT(S+ds, K, T, r, sigma)- BS_PUT(S,K,T,r,sigma))/ds
#     elif method== 'backward':
#         return (BS_PUT(S,K,T,r,sigma)- BS_PUT(S-ds, K , T, r, sigma))/ds
    
# K = 100
# r= 0.00
# T= 1
# sigma= 0.25
# S= 100

# prices= np.arange(1,250,1)

# deltas_c= delta_call(prices, K, T, r, sigma)
# deltas_p= delta_put(prices, K, T, r, sigma)
# deltas_back_c= delta_fdm_call(prices, K, T, r, sigma, ds= 0.01, method= 'backward')
# deltas_forward_p= delta_fdm_put(prices, K, T, r, sigma, ds= 0.01, method='forward')

# plt.plot(prices, deltas_c, label= 'Delta Call')
# plt.plot(prices, deltas_p, label='Delta Put')
# plt.xlabel('$S_0$')
# plt.ylabel('Delta')
# plt.title('Stock Price Effect on Delta for Calls/Puts')
# plt.axvline(K, color= 'black', linestyle= 'dashed', linewidth= 2, label="Strike")
# plt.legend()
# plt.show()


# # the curvature i.e. the second derivative is greatest near to where the option is in the money. 
# # The code below takes the difference between the analytic formula for delta and the finite difference approach. 

# errorc= np.array(deltas_c) - np.array(deltas_back_c)
# errorp= np.array(deltas_p) - np.array(deltas_forward_p)

# plt.plot(prices, errorc, label='FDM_CALL_ERROR')
# plt.plot(prices, errorp, label= 'FDM_PUT_ERROR')
# plt.legend()
# plt.xlabel('$S_0$')
# plt.ylabel('FDM Error')
# plt.legend()
# plt.show()








