# Import the library

import numpy as np
import qensfit.qensfit as qf


# Define a model function to fit the data. Here x is the independent
# variable, a, omega and phi are parameters, and q is a dataset-dependent
# constant. For more info about the function signature, please refer to
# the Model.target documentation

def fitmodel(x, /, a, omega, phi, *, q):
    """Test fit function"""
    return a * np.sin(omega * x + phi) * np.exp(-x / q)


# Declare a list of Parameter objects, with the SAME NAME as those
# you declared in the model function, so that you can control their initial
# values, bounds, fix them or make them global

lst = [qf.Parameter('omega', [0.2, 0.6, 1.7], 0., 50.,
                    ax_name = r'$\omega\ (rad/s)$'),
       qf.Parameter('phi', 1., 0., 5., is_global = True,
                    ax_name = r'$\Phi\ (rad)$'),
       qf.Parameter('a', 10., 5., 25., is_fixed = False),]


# Here we're generating some synthetic data as an example

x = np.array([np.linspace(0,20,201) for _ in range(3)])
y = np.zeros(x.shape)
y2 = np.zeros(x.shape)
dy = np.ones(x.shape) * 0.5
q = np.array([7.*(i+1) for i in range(3)])

for i in range(x.shape[0]):
    y[i] = (10. * (np.sin((i+1)/2 * x[i] + 1.) * np.exp(-x[i] / q[i])) +
            (1.5 * np.random.random_sample(y[i].shape) - 1))
    y2[i] = (15. * (np.sin((i+1)/2 * x[i] + 2.) * np.exp(-x[i] / q[i])) +
            (2. * np.random.random_sample(y[i].shape) - 1))


# Wrap the data in a dictionary containing QENSDataset objects (this
# will not be necessary if youare using the load_ascii function).

data = {'ds1': qf.QENSDataset(x = x, y = y, dy = dy, q = q),
        'ds2': qf.QENSDataset(x = x, y = y2, dy = dy, q = q)}


# Instantiate a Model object using the declared model function, the list
# of Parameter, and the input data dictionary. Then run the fit, et voilà!

mod = qf.Model(fitmodel, lst, data)
mod.run_fit(plot = True, autosave = True)