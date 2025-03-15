import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize



# Volatility Correction Term 
def vol_correction(sigma11=0.005,sigma22=0.003,sigma33=0.007,lambda_0=0.6,time=1):
    
    
    time = np.array(time, dtype=float)
    
    # Independent factor model
    sigma12 = 0
    sigma13 = 0
    sigma21 = 0
    sigma23 = 0
    sigma31 = 0
    sigma32 = 0
  
    Atilde = sigma11**2 + sigma12**2 + sigma13**2
    
    Btilde = sigma21**2+sigma22**2+sigma23**2
    
    Ctilde = sigma31**2+sigma32**2+sigma33**2
    
    Dtilde = sigma11*sigma21+sigma12*sigma22+sigma13*sigma23
    
    Etilde = sigma11*sigma31+sigma12*sigma32+sigma13*sigma33
    
    Ftilde = sigma21*sigma31+sigma22*sigma32+sigma23*sigma33
    
    Btilde = np.array(Btilde, dtype=float)
    Ctilde = np.array(Ctilde, dtype=float)
    Dtilde = np.array(Dtilde, dtype=float)
    Etilde = np.array(Etilde, dtype=float)
    Ftilde = np.array(Ftilde, dtype=float)

    res1 = Atilde*time**2/6.0
    
    res2 = Btilde*(1/(2*lambda_0**2)-(1-np.exp(-lambda_0*time))/(lambda_0**3*time)+(1-np.exp(-2*lambda_0*time))/(4*lambda_0**3*time))
    
    res3 = Ctilde*(1/(2*lambda_0**2)+np.exp(-lambda_0*time)/(lambda_0**2)-time*np.exp(-2*lambda_0*time)/(4*lambda_0)-3*np.exp(-2*lambda_0*time)/(4*lambda_0**2)-2*(1-np.exp(-lambda_0*time))/(lambda_0**3*time)+5*(1-np.exp(-2*lambda_0*time))/(8*lambda_0**3*time))
    
    res4 = Dtilde*(time/(2*lambda_0)+np.exp(-lambda_0*time)/(lambda_0**2)-(1-np.exp(-lambda_0*time))/(lambda_0**3*time))
    
    res5 = Etilde*(3*np.exp(-lambda_0*time)/(lambda_0**2)+time/(2*lambda_0)+time*np.exp(-lambda_0*time)/(lambda_0)-3*(1-np.exp(-lambda_0*time))/(lambda_0**3*time))
    
    res6 = Ftilde*(1/(lambda_0**2)+np.exp(-lambda_0*time)/(lambda_0**2)-np.exp(-2*lambda_0*time)/(2*lambda_0**2)-3*(1-np.exp(-lambda_0*time))/(lambda_0**3*time)+3*(1-np.exp(-2*lambda_0*time))/(4*lambda_0**3*time))
  
    return res1+res2+res3+res4+res5+res6

def g(lambda_0,mat):
    return 1/lambda_0 * (1-np.exp(-lambda_0*mat))


def NS_Loadings(lambda_0,bonds):
    bonds = np.array(bonds, dtype=float) 
    B = np.empty((len(bonds), 3))
    B[:,0] = 1
    B[:,1] = (1/bonds)*g(lambda_0,bonds)
    B[:,2] = (1/bonds)*(g(lambda_0,bonds)-bonds*np.exp(-lambda_0*bonds))
    return B


## This function returns the loglikelihood value which we want to optimize.
def Kalman_filter(para, number_of_obs, number_of_mats, maturities, observed_data): #para is a vector containing our set of parameters

    
    # set dt as monthly
    dt = 1/12
  
    # initialize all the parameter values.
    kappa11=para[0]
    kappa22=para[1]
    kappa33=para[2]
  
    theta1=para[3]
    theta2=para[4]
    theta3=para[5]
  
  # Force positive
    sigma_1=np.abs(para[6])
    sigma_2=np.abs(para[7])
    sigma_3=np.abs(para[8])
  
    lambda_0 = para[9]
  
    # Note the squared, I.e. our parameter is the standard deviation but we need the variance matrix, so we square it here. Also ensures positivity
    sigma_err_sq=para[10]**2 
  
    # Initialize K^P, \theta^P, H, Sigma
    K = np.diag([kappa11,kappa22,kappa33])
  
    theta=np.array([theta1,theta2,theta3])

    
    Sigma=np.diag([sigma_1,sigma_2,sigma_3])
  
    H = np.diag([sigma_err_sq] * number_of_mats) # number of observations per observation date.
  
    # Impose stationarity - I.e. check that eigenvalues of K^P are positive
    # If not let the function return some large value e.g. 999999
    # Save the eigenvalues and vectors of K^P.
  
    eigenvalues, eigenvectors = np.linalg.eig(K)

    eigenvalues = np.array(eigenvalues)
    # Check for stationarity condition
    if np.min(np.real(eigenvalues)) < 0 or lambda_0 < 0:
        print("stationarity")
        return 999999

    
    # Calculate the A and B used in the measurement equation.
  
    B=NS_Loadings(lambda_0,maturities)
  
    A=vol_correction(sigma_1,sigma_2,sigma_3,lambda_0,maturities)
  

    # We calculate the conditional and unconditional covariance matrix

    # Step 1: Calculate the S-overline matrix  
    InvEigenvectors = np.array(np.linalg.solve(eigenvectors, np.eye(3)))   
    
    # Compute Smat
    Smat = InvEigenvectors @ Sigma @ Sigma.T @ InvEigenvectors.T
    
  
    # Step 2: Calculate the V-overline matrix for dt and in the limit
    Vmat = np.zeros((3, 3))
    Vlim = np.zeros((3, 3))
    i=0
    while(i<3):
        j=0
        while(j<3):
    
            Vmat[i,j]=Smat[i,j]*(1-np.exp(-(eigenvalues[i]+eigenvalues[j])*dt))/(eigenvalues[i]+eigenvalues[j])
            Vlim[i,j]=Smat[i,j]/(eigenvalues[i]+eigenvalues[j])
            j=j+1
    
        i=i+1


  

  # Step 3: Calculate the final analytical covariance matrices
  #Take real parts
    
    Q = np.real(eigenvectors @ Vmat @ eigenvectors.T)
    Q_unconditional = np.real(eigenvectors @ Vlim @ eigenvectors.T)

    
  
  # Start the filter at the unconditional mean and variance.

    X=theta
  
    P = Q_unconditional

  ## Calculate F_t and C_t
  
    K_dt=-K*dt

    Ft=expm(K_dt)
    
    
   
    Ct=(np.eye(3)-Ft) @ theta

    #Set the initial log likelihood value to zero.
    loglike=0

    rec_X = np.empty((3,number_of_obs)) # antal faktorer, antal rækker
    rec_y=np.empty((number_of_mats,number_of_obs)) 
    
    # Iterate over all the observation dates
    i=0
    totalNo=number_of_obs #totalNO is the number of observation dates
    
    while(i<totalNo): 
     
      # Perform the prediction step
      X_pred=Ct+Ft @X
      P_pred=Ft@P@Ft.T+Q
      
      # calculate the model-implied yields
      
      yimplied=A+B@X_pred
      
      ## Calculate the prefit residuals based on the observed and implied yields
      
      y = observed_data[i,:].astype(float)
      
      v=y-yimplied

      #Calculate the covariance matrix of the prefit residuals

      S=B@P_pred@B.T+H
      
      # Calculate the determinant of the covariance matrix of the prefit residuals
      # Check that the determinant is defined, finite, and positive.
      # If not let the function return some large value e.g. 8888888

      detS=np.linalg.det(S)
      
      if np.isnan(detS) or np.isinf(detS) or detS < 0:
        print("determinant")
        return 888888
      
      # Calculate the log determinant
      
      logdetS=np.log(detS)
      
      # Calculate the inverse of the covariance matrix

      Sinv = np.linalg.solve(S, np.eye(len(y)))

      # Perform the update step

      X=X_pred+P_pred@B.T@Sinv@v
      P=P_pred-P_pred@B.T@Sinv@B@P_pred
    
      
      rec_X[:,i] = X
      rec_y[:,i] = A+B@X

      
      #Calculate the ith log likelihood contribution and add the ith log likelihood contribution to the total log likelihood value

      loglike_cont = (-0.5 * len(y) * np.log(2 * np.pi) - 0.5 * logdetS - 0.5 * (v.T @ Sinv @ v))
      
      loglike += loglike_cont
      
      
      ## Adding 1 to iteration count
      i=i+1
    
  #Returning the negative log likelihood (assuming we are minimizing)
    print(loglike) 
    return -(loglike)



## This function returns the loglikelihood value which we want to optimize.
def Kalman_filter_optimized(para, number_of_obs, number_of_mats, maturities, observed_data): #para is a vector containing our set of parameters

    
    #set dt as monthly
    dt = 1/12
  
    # initialize all the parameter values.
    kappa11=para[0]
    kappa22=para[1]
    kappa33=para[2]
  
    theta1=para[3]
    theta2=para[4]
    theta3=para[5]
  
  #Force positive
    sigma_1=np.abs(para[6])
    sigma_2=np.abs(para[7])
    sigma_3=np.abs(para[8])
  
    lambda_0 = para[9]
  
    #Note the squared, I.e. our parameter is the standard deviation but we need the variance matrix, so we square it here. Also ensures positivity
    sigma_err_sq=para[10]**2 
  
    # Initialize K^P, \theta^P, H, Sigma
    K = np.diag([kappa11,kappa22,kappa33])
  
    theta=np.array([theta1,theta2,theta3])

    
    Sigma=np.diag([sigma_1,sigma_2,sigma_3])
  
    H = np.diag([sigma_err_sq] * number_of_mats) #number of observations per observation date.
  
    # Impose stationarity - I.e. check that eigenvalues of K^P are positive
    # If not let the function return some large value e.g. 999999
    # Save the eigenvalues and vectors of K^P.
  
    eigenvalues, eigenvectors = np.linalg.eig(K)

    eigenvalues = np.array(eigenvalues)
    # Check for stationarity condition
    if np.min(np.real(eigenvalues)) < 0 or lambda_0 < 0:
        print("stationarity")
        return 999999

    
    # Calculate the A and B used in the measurement equation. 
  
    B=NS_Loadings(lambda_0,maturities)
  
    A=vol_correction(sigma_1,sigma_2,sigma_3,lambda_0,maturities)
  

    # We calculate the conditional and unconditional covariance matrix

    # Step 1: Calculate the S-overline matrix  
    InvEigenvectors = np.array(np.linalg.solve(eigenvectors, np.eye(3)))   
    
    # Compute Smat
    Smat = InvEigenvectors @ Sigma @ Sigma.T @ InvEigenvectors.T
    
  
    # Step 2: Calculate the V-overline matrix for dt and in the limit
    Vmat = np.zeros((3, 3))
    Vlim = np.zeros((3, 3))
    i=0
    while(i<3):
        j=0
        while(j<3):
    
            Vmat[i,j]=Smat[i,j]*(1-np.exp(-(eigenvalues[i]+eigenvalues[j])*dt))/(eigenvalues[i]+eigenvalues[j])
            Vlim[i,j]=Smat[i,j]/(eigenvalues[i]+eigenvalues[j])
            j=j+1
    
        i=i+1


  

  # Step 3: Calculate the final analytical covariance matrices
  #Take real parts
    
    Q = np.real(eigenvectors @ Vmat @ eigenvectors.T)
    Q_unconditional = np.real(eigenvectors @ Vlim @ eigenvectors.T)

    
  
  # Start the filter at the unconditional mean and variance.

    X=theta
  
    P = Q_unconditional

  ## Calculate F_t and C_t
  
    K_dt=-K*dt

    Ft=expm(K_dt)
    
    
   
    Ct=(np.eye(3)-Ft) @ theta

    #Set the initial log likelihood value to zero.
    loglike=0

    rec_X = np.empty((3,number_of_obs)) # antal faktorer, antal rækker
    rec_y=np.empty((number_of_mats,number_of_obs)) 
    
    # Iterate over all the observation dates
    i=0
    totalNo=number_of_obs #totalNO is the number of observation dates in your data
    
    while(i<totalNo): 
     
      # Perform the prediction step
      X_pred=Ct+Ft @X
      P_pred=Ft@P@Ft.T+Q
      
      # calculate the model-implied yields
      
      yimplied=A+B@X_pred
      
      ## Calculate the prefit residuals based on the observed and implied yields
      
      y = observed_data[i,:].astype(float)
      
      v=y-yimplied

      #Calculate the covariance matrix of the prefit residuals

      S=B@P_pred@B.T+H
      
      # Calculate the determinant of the covariance matrix of the prefit residuals
      # Check that the determinant is defined, finite, and positive.
      # If not let the function return some large value e.g. 8888888

      detS=np.linalg.det(S)
      
      if np.isnan(detS) or np.isinf(detS) or detS < 0:
        print("determinant")
        return 888888
      
      # Calculate the log determinant
      
      logdetS=np.log(detS)
      
      # Calculate the inverse of the covariance matrix

      Sinv = np.linalg.solve(S, np.eye(len(y)))

      # Perform the update step

      X=X_pred+P_pred@B.T@Sinv@v
      P=P_pred-P_pred@B.T@Sinv@B@P_pred
    
      
      rec_X[:,i] = X 
      rec_y[:,i] = A+B@X 

      
      #Calculate the ith log likelihood contribution and add the ith log likelihood contribution to the total log likelihood value

      loglike_cont = (-0.5 * len(y) * np.log(2 * np.pi) - 0.5 * logdetS - 0.5 * (v.T @ Sinv @ v))
      
      loglike += loglike_cont
      
      
      ## Adding 1 to iteration count
      i=i+1
    
  #Returning the negative log likelihood (assuming we are minimizing)
    return rec_X, rec_y

