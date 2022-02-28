# bayesian-GLM
Bayesian GLMs! The way to think about these is a more generalized version ofany given linear model, and it neatly wraps logistic regression, count regression, and poisson regression into one model that can do it all. In this project:
- the A and usps datasets correspond to logistic regression
- the AP dataset corresponds to poisson regression
- the AO dataset corresponds to ordinal regression

Check out this [link](https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab) for a more in depth explanation of GLMs!

## Results
For each of these datasets, I set aside 1/3 of the data for the test set. I then iterated through using different portions of the training data to train and then subsequently test our model and plotted error, # of iterations until convergence, and run time all as a function of training set size.

### Logistic Regression (A and usps datasets)
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/A.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/A_iterations.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/A_runtime.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/usps.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/usps_iterations.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/usps_runtime.png?raw=true" alt="drawing" width="350"/>



### Count/Poisson Regression (AP dataset)
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AP.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AP_iterations.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AP_runtime.png?raw=true" alt="drawing" width="350"/>



### Ordinal Regression (AO dataset)
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AO.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AO_iterations.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/bayesian-GLM/blob/main/images/AO_runtime.png?raw=true" alt="drawing" width="350"/>
