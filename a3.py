import numpy as np
import pandas as pd
import random
import math
import time
import matplotlib.pyplot as plt

# classification w logistic regression
A = pd.read_csv('pp3data/A.csv').to_numpy()
A_labels = pd.read_csv('pp3data/labels-A.csv').to_numpy()
usps = pd.read_csv('pp3data/usps.csv').to_numpy()
usps_labels = pd.read_csv('pp3data/labels-usps.csv').to_numpy()

# count prediction with Poisson regression
AP = pd.read_csv('pp3data/AP.csv').to_numpy()
AP_labels = pd.read_csv('pp3data/labels-AP.csv').to_numpy()

# ordinal prediction with ordinal regression
AO = pd.read_csv('pp3data/AO.csv').to_numpy()
AO_labels = pd.read_csv('pp3data/labels-AO.csv').to_numpy()

# idk
irlstest = pd.read_csv('pp3data/irlstest.csv').to_numpy()
irlstest_labels = pd.read_csv('pp3data/labels-irlstest.csv').to_numpy()
irlsw = pd.read_csv('pp3data/irlsw.csv').to_numpy()





# let's get started
class GLM():
    # model = logistic/poisson/ordinal
    def __init__(self, data, labels, model, t_size):
        self.model = model

        start = time.time()
        train_i, test_i = self.train_test_split(data, t_size)
        self.train_phi = np.concatenate([np.ones([len(train_i), 1]), data[train_i]], axis=1)
        self.train_t = labels[train_i].reshape(len(train_i))
        self.test_phi = np.concatenate([np.ones([len(test_i), 1]), data[test_i]], axis=1)
        self.test_t = labels[test_i].reshape(len(test_i))

        self.N = len(self.test_phi)

        if model == 'ordinal':
            self.ordinal_phis = [-math.inf, -2, -1, 0, 1, math.inf]


        self.w_new = np.zeros(len(self.train_phi[0]))
        self.w_old = self.w_new + 100
    
        self.alpha = 10

        i = 0
        while not self.converged():
            if i == 100:
                break
            y = self.train_predict()
            self.update_weights(y)
            i +=1

        w_map = self.w_new 
        yhat = self.test_predict(w_map)


        self.err = self.error(yhat)
        self.duration = time.time() - start
        self.iterations = i+1


    def train_test_split(self, data, t_size):
        indexes = list(range(len(data)))
        random.shuffle(indexes)

        cutoff = int(2/3*len(indexes))
        train_i = indexes[:int(t_size*cutoff)]
        test_i = indexes[cutoff:]

        return train_i, test_i

    def sigmoid(self, a):
        return 1/(1+np.exp(-a))


    def train_predict(self):
        if self.model == 'logistic':
            return self.sigmoid(np.matmul(self.train_phi, np.transpose(self.w_new)))

        elif self.model == 'poisson':
            return np.exp(np.matmul(self.train_phi, np.transpose(self.w_new)))

        elif self.model == 'ordinal':
            ys = []

            # should be Nx1
            a = np.matmul(self.train_phi, np.transpose(self.w_new))
            for a_i in a:
                temp = []
                for p in self.ordinal_phis:
                    temp.append(self.sigmoid(p-a_i))
                ys.append(temp)

            return np.array(ys) # ask about this


    def d(self, y):
        if self.model == 'logistic' or self.model == 'poisson':
            return self.train_t - y

        elif self.model == 'ordinal':
            ds = []
            for i in range(len(y)):
                curr_t = self.train_t[i]
                ds.append(y[i][curr_t] + y[i][curr_t-1] - 1)

            return np.array(ds) # ask about this
    
    def r(self, y):
        if self.model == 'logistic':
            return np.multiply(y, (1-y))
        elif self.model == 'poisson':
            return y
        elif self.model == 'ordinal':
            rs = []
            for i in range(len(y)):
                curr_t = self.train_t[i]
                rs.append(y[i][curr_t]*(1-y[i][curr_t]) + y[i][curr_t-1]*(1-y[i][curr_t-1]))

            return np.array(rs)


    def first_derivative(self, d):
        return np.subtract(np.matmul(np.transpose(self.train_phi), np.transpose(d)), self.w_new*self.alpha)
    
    def second_derivative(self, r):
        # might need to transpose r before diagonalizing
        return -np.matmul(np.matmul(np.transpose(self.train_phi), np.diag(r)), self.train_phi) - self.alpha*np.identity(len(self.train_phi[0]))
    

    def update_weights(self, y):
        self.w_old = self.w_new
        # is np.matmul appropriate here?
        d = self.d(y)
        g = self.first_derivative(d)
        r = self.r(y)
        H = self.second_derivative(r)

        self.w_new = self.w_old - np.matmul(np.linalg.inv(H), g)

    def converged(self):
        try:
            if np.linalg.norm(np.subtract(self.w_new, self.w_old))/np.linalg.norm(self.w_old) < .001:
                return True
            else:
                return False
        except RuntimeWarning:
            # I was getting dividing by zero errors, in which case the val is at infinity and we
            # haven't converged
            return False


    def test_predict(self, w):
        if self.model == 'logistic':
            return self.sigmoid(np.matmul(self.train_phi, np.transpose(self.w_new)))
        elif self.model == 'poisson':
            a = np.matmul(self.test_phi, np.transpose(w))
            lam = np.exp(a)
            return [int(i) for i in lam]
        elif self.model == 'ordinal':
            ys = []

            # should be Nx1
            a = np.matmul(self.test_phi, np.transpose(self.w_new))
            for a_i in a:
                temp = []
                for p in self.ordinal_phis:
                    temp.append(self.sigmoid(p-a_i))
                ys.append(temp)
            
            yhat = []
            for i in range(len(ys)):
                ps = []
                for j in range(1, len(ys[i])):
                    ps.append(ys[i][j] - ys[i][j-1])

                yhat.append(1 + ps.index(max(ps)))

            return yhat 


    def error(self, yhat):
        if self.model == 'logistic':
            return sum([1 if y!=t else 0 for y,t in zip(yhat, self.test_t)])/self.N
        elif self.model == 'poisson':
            return sum([abs(y-t) for y,t in zip(yhat, self.test_t)])/self.N
        elif self.model == 'ordinal':
            return sum(abs(yhat-self.test_t))/self.N
    


data = {
    'A': ['logistic', A, A_labels],
    'usps': ['logistic', usps, usps_labels],
    'AP': ['poisson', AP, AP_labels],
    'AO': ['ordinal', AO, AO_labels]
}

training_sizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

results_by_dataset = []
times_by_dataset = []
for dataset in data.keys():
    results_by_t_size = []
    time_by_t_size = []
    for t_size in training_sizes:
        errors = []
        iters = []
        durations = []
        for _ in range(30):
            model = data[dataset][0]
            inputs = data[dataset][1]
            labels = data[dataset][2]

            res = GLM(inputs, labels, model, t_size)
            errors.append(res.err)
            iters.append(res.iterations)
            durations.append(res.duration)

        #print(f'errors {errors}')
        results_by_t_size.append([np.mean(errors), np.std(errors), np.mean(iters), np.mean(durations)])
    
    results_by_dataset.append(results_by_t_size)


                

results = {
    'A': results_by_dataset[0],
    'usps': results_by_dataset[1],
    'AP': results_by_dataset[2],
    'AO': results_by_dataset[3]
}



for result in results.keys():
    plt.title(result)

    res = results[result]
    
    means = []
    stds = []
    iters = []
    durations = []

    for t in range(len(res)):
        means.append(res[t][0])
        stds.append(res[t][1])
        iters.append(res[t][2])
        durations.append(res[t][3])


    plt.errorbar(y=means, x=training_sizes, yerr=stds)
    plt.xlabel('training_set_size')
    plt.ylabel('error: mean and s.d.')
    plt.show() 

    # duration by t set size plot
    plt.title(result + ' runtime by training_set_size')
    plt.plot(training_sizes, durations)
    plt.xlabel('training_set_size')
    plt.ylabel('duration')
    plt.show() 

    # iterations by t set size plot
    plt.title(result + ' iterations by training_set_size')
    plt.plot(training_sizes, iters)
    plt.xlabel('training_set_size')
    plt.ylabel('iterations')
    plt.show() 
    