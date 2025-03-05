# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:09:16 2024

@author: Antonio Caliò

A little library that enables easy fitting of Quasielastic
Neutron Scattering data.
It borrows some concept from the popular lmfit package,
such as the Model and Parameter objects, but everything is
implemented from scratch to make it compatible with
Global Fitting procedures, which are essential for QENS data.

USAGE:

- Use LoadAscii to load data reduced with Mantid, or load your data another
way and then organise it in a dictionary of QENSDataset instances.
Loading of multiple datasets is supported.

- Define your model function to fit the data. Constants are also supported.

- Declare a list of Parameter obejcts to control their initial value, bounds,
and whether the parameter is fixed and/or global.

- Declare a model instance using the function you wrote before as the target,
and feed it the list of Parameter objects and the QENSDataset.

- Run the fit.

- Plot fits and parameters, and Save your results.
"""

import inspect
import glob
import os
import re
import copy
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from alive_progress import alive_bar
from subplot_cycler import SubplotCycler

class Parameter:
    """
    Object that basicaly reimplements a float (with all its implicit
    methods) with added attributes to make it fit-compatible. That is,
    all the operations are defined to be applied on Parameter.value.
    Moreover, it supports indexing when the value is an array-like.
    """
    def __init__(self,
                 name: str,
                 value_0: float or list or np.ndarray,
                 low_bound: float or list or np.ndarray = None,
                 high_bound: float or list or np.ndarray = None,
                 is_fixed: bool = False,
                 is_global: bool = False,
                 ax_name: str = None):
        """
        Initialises the Parameter instance

        Parameters
        ----------
        name : str
            Name of the parameter.
        value_0 : float or list or np.ndarray
            Initial value.
        low_bound : float or list or np.ndarray, optional
            Lower bound. The default is None, which will imply an
            unbounded parameter
        high_bound : float or list or np.ndarray, optional
            Upper bound. The default is None, which will imply an
            unbounded parameter
        is_fixed : bool, optional
            Used to fix a parameter. The default is False.
        is_global : bool, optional
            Used to make a parameter global. The default is False.
        ax_name : str, optional
            String that will appear as axis label when the parameter is
            plotted, supports LaTeX. The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        self.ax_name = ax_name if ax_name is not None else name
        self.ini = value_0
        self.is_fixed = is_fixed

        if self.is_fixed:
            if isinstance(self.ini, list):
                self.ini = np.array(self.ini)
            self.low = self.ini - 1e-9
            self.high = self.ini + 1e-9
        else:
            if low_bound is None:
                self.low = -np.inf
            else:
                self.low = low_bound

            if high_bound is None:
                self.high = np.inf
            else:
                self.high = high_bound

        self.is_global = is_global
        self.is_free = (not is_fixed) and (not is_global)
        self.value = 0.
        self.error = 0.

    def __getitem__(self, index):
        if isinstance(self.value, (list, np.ndarray)):
            return self.value[index]
        else:
            return self.value

    def __setitem__(self, index, other):
        if isinstance(self.value, (list, np.ndarray)):
            self.value[index] = other
            return self
        else:
            self.value = other
            return self

    def __repr__(self):
        return (f"Parameter: {self.name}, Value = {self.value} +/- " +
                f"{self.error}, Initial = {self.ini}, Bounds = " +
                f"{self.low} - {self.high}, Fixed = {self.is_fixed}, " +
                f"Global = {self.is_global}\n")

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __truediv__(self, other):
        return self.value / other

    def __pow__(self, other):
        return self.value ** other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rmul__(self, other):
        return other * self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __rpow__(self, other):
        return other ** self.value

    def __iadd__(self, other):
        self.value = self.value + other
        return self

    def __isub__(self, other):
        self.value = self.value - other
        return self

    def __imul__(self, other):
        self.value = self.value * other
        return self

    def __itruediv__(self, other):
        self.value = self.value / other
        return self

    def __ipow__(self, other):
        self.value = self.value ** other
        return self

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __is___(self, other):
        return self.value is other

    def __is_not__(self, other):
        return self.value is not other

class ParList:
    """
    Parameter List object. Contains the Parameters in both list and dictionary
    formats, a list of parameter names, number of free, fixed or global
    parameters. The ParList object itself can be indexed both like an array
    (requires knowledge of the order of paramters), and like a dictionary
    (using parameter names as the key).
    All the pack and unpack functions shouldn't be needed, as they're only
    used when curve_fit is called, as it only accepts 1D inputs.
    """
    def __init__(self,
                 parlist: list = None,
                 curves: int = 1):
        """
        Initialize the ParList instance.

        Parameters
        ----------
        parlist : list, optional
            List containing the Parameter objects to add. The default is None.
        curves : int, optional
            Number of curves to be fitted simultaneously using those
            parameters. The default is 1.

        Returns
        -------
        None.

        """
        self.list = parlist
        self.n_curves = curves
        self._init_pars()
        self.dict = self._make_dict()
        self.names = self._get_names()
        self.n_free = self._get_n_free()
        self.n_fixed = self._get_n_fixed()
        self.n_global = self._get_n_global()
        self.n_total = ((self.n_curves * self.n_free) +
                        (self.n_curves * self.n_fixed) +
                        self.n_global)
        self.values = self.unpack_values()
        self.errors = self.unpack_errors()

    def __repr__(self):
        return (f"Parameter List: {self.n_free} free, {self.n_fixed} fixed, " +
                f"{self.n_global} global. {self.n_total} total parameters " +
                f"for {self.n_curves} curves\n")

    def __str__(self):
        return ''.join([str(i) for i in self.list])

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.dict[index]
        else:
            return self.list[index]

    def _init_pars(self):
        for par in self.list:
            setattr(self, par.name, par)
            if par.is_global:
                par.value = 0.
                par.error = 0.
            else:
                par.value = np.zeros(self.n_curves)
                par.error = np.zeros(self.n_curves)

    def _make_dict(self):
        pardict = {}
        for par in self.list:
            pardict[par.name] = par
        return pardict

    def _get_names(self):
        names = []
        for par in self.list:
            names.append(par.name)
        return names

    def _get_n_free(self):
        n = 0
        for par in self.list:
            if (not par.is_fixed) and (not par.is_global):
                n = n+1
        return n

    def _get_n_fixed(self):
        n = 0
        for par in self.list:
            if par.is_fixed:
                n = n+1
        return n

    def _get_n_global(self):
        n = 0
        for par in self.list:
            if par.is_global:
                n = n+1
        return n

    def _update(self):
        self.names = self._get_names()
        self.dict = self._make_dict()
        self.n_free = self._get_n_free()
        self.n_fixed = self._get_n_fixed()
        self.n_global = self._get_n_global()
        self.n_total = ((self.n_curves *self.n_free) +
                        (self.n_curves * self.n_fixed) +
                        self.n_global)
        self.values = self.unpack_values()
        self.errors = self.unpack_errors()

    def add_par(self, par):
        """
        Adds a parameter to the ParList instance.

        Parameters
        ----------
        par : TYPE
            Parameter object to be added (required).

        Returns
        -------
        None.

        """
        if par.is_global:
            par.value = 0.
            par.error = 0.
        else:
            par.value = np.zeros(self.n_curves)
            par.error = np.zeros(self.n_curves)
        self.list.append(par)
        self._update()

    def unpack_pin(self) -> np.ndarray:
        """
        Unpacks the initial values of all the parameters,
        and stores them in a 1D numpy.ndarray

        Raises
        ------
        RuntimeError
            The initial value must be int or float (for global parameters),
            list or numpy.ndarray (for free parameters).

        Returns
        -------
        numpy.ndarray
            1D array containing all the initial values.

        """
        pin = []
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    if isinstance(par.ini, list):
                        pin.append(par.ini[i])
                    elif isinstance(par.ini, np.ndarray):
                        pin.append(par.ini[i])
                    elif isinstance(par.ini, float):
                        pin.append(par.ini)
                    elif isinstance(par.ini, int):
                        pin.append(par.ini)
                    else:
                        raise RuntimeError('Initial value must be float,' +
                                             ' list or array')
        for par in self.list:
            if par.is_global:
                pin.append(par.ini)
        return np.array(pin)

    def unpack_bounds(self) -> (np.ndarray, np.ndarray):
        """
        Unpacks the bounds of all the parameters,
        and stores them in a tuple of two 1D numpy.ndarray

        Raises
        ------
        RuntimeError
            The bounds must be int or float (for global parameters),
            list or numpy.ndarray (for free parameters).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            tuple of two 1D arrays containing the upper and lower bounds.

        """
        low = []
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    if isinstance(par.low, (list, np.ndarray)):
                        low.append(par.low[i])
                    elif isinstance(par.low, (int, float)):
                        low.append(par.low)
                    else:
                        raise RuntimeError('Bounds must be int, float,' +
                                             ' list or array')
        for par in self.list:
            if par.is_global:
                low.append(par.low)
        high = []
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    if isinstance(par.high, (list, np.ndarray)):
                        high.append(par.high[i])
                    elif isinstance(par.high, (int, float)):
                        high.append(par.high)
                    else:
                        raise RuntimeError('Bounds must be int, float,' +
                                             ' list or array')
        for par in self.list:
            if par.is_global:
                high.append(par.high)

        return (np.array(low), np.array(high))

    def unpack_values(self) -> np.ndarray:
        """
        Unpacks the values of all the parameters,
        and stores them in a 1D numpy.ndarray. If no fitting has been
        attempted yet, all values will be equal to the initial values.

        Returns
        -------
        numpy.ndarray
            1D array containing all the parameter values.

        """
        val = []
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    val.append(par.value[i])
        for par in self.list:
            if par.is_global:
                val.append(par.value)
        return np.array(val)

    def unpack_errors(self) -> np.ndarray:
        """
        Unpacks the uncertainties of all the parameters,
        and stores them in a 1D numpy.ndarray. If no fitting has been
        attempted yet, all values will be None.

        Returns
        -------
        numpy.ndarray
            1D array containing all the uncertainties values.

        """
        err = []
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    err.append(par.error[i])
        for par in self.list:
            if par.is_global:
                err.append(par.error)
        return np.array(err)

    def pack_values(self, vector: np.ndarray):
        """
        Packs the provided values into the Parameter objects into the ParList.
        WARNING: don't use if you don't know EXACTLY the order of the
        parameters!

        Parameters
        ----------
        vector : numpy.ndarray
            Array containing the values to be packed.

        Returns
        -------
        None.

        """
        self.values = vector
        l = 0
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    par.value[i] = vector[l]
                    l = l+1
        for par in self.list:
            if par.is_global:
                par.value = vector[l]
                l = l+1
        self._make_dict()

    def pack_errors(self, vector: np.ndarray):
        """
        Packs the provided uncertaintiesinto the Parameter objects
        into the ParList.
        WARNING: don't use if you don't know EXACTLY the order of the
        parameters!

        Parameters
        ----------
        vector : numpy.ndarray
            Array containing the uncertainties to be packed.

        Returns
        -------
        None.

        """
        self.errors = vector
        l = 0
        for i in range(self.n_curves):
            for par in self.list:
                if not par.is_global:
                    par.error[i] = vector[l]
                    l = l+1
        for par in self.list:
            if par.is_global:
                par.error = vector[l]
                l = l+1
        self._make_dict()

    def free_to_df(self, index = None, index_title = None) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame containing the values and uncertainties
        for all the free parameters.
        The index and its title can be customised.

        Parameters
        ----------
        index : TYPE, optional
            Index values for the DataFrame. The default is None.
        index_title : TYPE, optional
            Index title for the DataFrame. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the values and uncertainties
            for all the free parameters as columns.

        """
        cols = []
        if index is None:
            index = range(1, self.n_curves + 1)
        for par in self.list:
            if par.is_free:
                cols.extend((par.name, 'err_' + par.name))
        df = pd.DataFrame(data = None,
                          columns = cols,
                          index = index)
        for par in self.list:
            if par.is_free:
                df[par.name] = par.value
                df['err_' + par.name] = par.error
        df.index.name = index_title
        return df

    def all_to_df(self, index = None, index_title = None) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame containing the values and uncertainties
        for all the parameters (free and global).
        The index and its title can be customised.

        Parameters
        ----------
        index : TYPE, optional
            Index values for the DataFrame. The default is None.
        index_title : TYPE, optional
            Index title for the DataFrame. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the values and uncertainties
            for all the parameters (free and global) as columns.

        """
        cols = []
        if index is None:
            index = range(1, self.n_curves + 1)
        for par in self.list:
            cols.extend((par.name, 'err_' + par.name))
        df = pd.DataFrame(data = None,
                          columns = cols,
                          index = index)
        for par in self.list:
            df[par.name] = par.value
            df['err_' + par.name] = par.error
        df.index.name = index_title
        return df

    def global_to_df(self, index = [1], index_title = None) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame containing the values and uncertainties
        for the global parameters. This will have different dimensions than
        the DataFrame returned by all_to_df() or free_to_df(), as each global
        parameter will correspond to multiple curves.
        The index and its title can be customised.

        Parameters
        ----------
        index : TYPE, optional
            Index values for the DataFrame. The default is [1].
        index_title : TYPE, optional
            Index title for the DataFrame. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the values and uncertainties
            for the global parameters as columns.

        """
        cols = []
        for par in self.list:
            if par.is_global:
                cols.extend((par.name, 'err_' + par.name))
        df = pd.DataFrame(data = None,
                          columns = cols,
                          index = index)
        for par in self.list:
            if par.is_global:
                df[par.name] = par.value
                df['err_' + par.name] = par.error
        df.index.name = index_title
        return df

    def all_to_df_noerr(self) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame containing the values
        for all the parameters (free and global).
        No uncertainties are provided!

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the values
            for all the parameters (free and global) as columns.
            No uncertainties are provided!

        """
        df = pd.DataFrame(data = None,
                          columns = self.names,
                          index = range(1, self.n_curves + 1))
        for par in self.list:
            df[par.name] = par.value
        return df

class QENSDataset:
    """
    Dataset object which can be instantiated by itself, provided with
    suitable data, or it can be automatically created by the load_ascii
    function. Data is checked automatically for dimension mismatches.
    """
    def __init__(self,
                 name: str = '_',
                 x: list or np.ndarray = None,
                 y: list or np.ndarray = None,
                 dy: list or np.ndarray = None,
                 q: list or np.ndarray = None):
        """
        Initialise the QENSDataset instance.

        Parameters
        ----------
        name : str, optional
            Dataset name. The default is '_'.
        x : list or np.ndarray, optional
            Dataset x axis, i.e. the Energy axis.
            Has to have the same dimensions as y and dy.
            The default is None.
        y : list or np.ndarray, optional
            Dataset y axis, i.e. the Scattering Intensity axis.
            Has to have the same dimensions as x and dy.
            The default is None.
        dy : list or np.ndarray, optional
            Dataset dy axis, i.e. the Errors axis.
            Has to have the same dimensions as x and yy.
            The default is None.
        q : list or np.ndarray, optional
            Dataset q axis, i.e. the Momentum Transfer axis.
            If x, y and dy have dimensions (N, M), then
            q has to be 1D and of length N.
            The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        self.x = np.array(x)
        self.y = np.array(y)
        self.dy = np.array(dy)
        self.q = np.array(q)
        self.n_q = len(self.q)
        self.n_e = len(self.x[0])
        self._check_data()

    def __repr__(self):
        return (f'QENS Dataset {self.name}, Dimensions: ({self.n_q} spectra, '
                f'{self.n_e} energies), Range: {self.x[0,0]} meV - '
                f'{self.x[0,-1]} meV\n')

    def _check_data(self):
        if not all(self.x.shape == i.shape
                   for i in [self.x, self. y, self.dy]):
            raise RuntimeError('Dimension mismatch: '
                               f'x: {self.x.shape}, y: {self.y.shape}, '
                               f'errors: {self.dy.shape}.')
        if not all(self.n_q == i.shape[0]
                   for i in [self.x, self. y, self.dy]):
            raise RuntimeError(f'Dimension mismatch: {self.n_q} q values, but '
                               f'{self.x.shape[0]} x vectors, '
                               f'{self.y.shape[0]}, spectra '
                               f'and {self.dy.shape[0]} error vectors.')

class QENSResult:
    """
    Contains the results of the fit. It is instantiated bu Model.run_fit()
    when a dataset is fitted.
    WARNING: If this object is used outside its intended scope,
    the validate_result() method has to be called by hand!
    """
    def __init__(self,
                 name: str = '_',
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 params: ParList = None,
                 chisq: float = None,
                 popt: np.ndarray = None,
                 pcov: np.ndarray = None,
                 infodict: dict = None,
                 mesg: str = None,
                 ier: int = None,
                 residuals: np.ndarray = None):
        """
        Initialise the QENSResult instance.

        Parameters
        ----------
        name : str, optional
            Result name. Usually the same as the QENSDataset it refers to.
            The default is '_'.
        x : np.ndarray, optional
            Result x axis. This array is denser than the input data (i.e.
            QENSDataset.x) to make the plots look smoother.
            The default is None.
        y : np.ndarray, optional
            Result y axis, i.e. the model function evaluated at the best fit
            parameter values. The default is None.
        params : ParList, optional
            ParList instance containing the best fit parameters.
            The default is None.
        chisq : float, optional
            Chi squared value for the global fit. The default is None.
        popt : np.ndarray, optional
            1D vector containing the values of the best fit parameters.
            See scipy.optimize.curve_fit for more info. The default is None.
        pcov : np.ndarray, optional
            2D vector containing approximate covariance of popt.
            See scipy.optimize.curve_fit for more info. The default is None.
        infodict : dict, optional
            Dictionary containing the keys 'nfev' (number of function
            evaluations) and 'fvec' (Residuals evaluated at the solution
            in a 1D array).
            See scipy.optimize.curve_fit for more info. The default is None.
        mesg : str, optional
            A string message giving information about the solution.
            See scipy.optimize.curve_fit for more info.. The default is None.
        ier : int, optional
            An integer flag. If it is equal to 1, 2, 3 or 4, the solution
            was found. Otherwise, the solution was not found..
            See scipy.optimize.curve_fit for more info. The default is None.
        residuals : np.ndarray, optional
            Residuals (infodict['fvec']) reshaped to have the same dimensions
            of the y axis. The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        self.x = x
        self.y = y
        self.params = params
        self.chisq = chisq
        self.popt = popt
        self.pcov = pcov
        self.infodict = infodict
        self.mesg = mesg
        self.ier = ier
        self.residuals = residuals
        self.cycler = None

    def __repr__(self):
        if self.ier in [1, 2, 3, 4]:
            converged = 'Yes'
        else:
            converged = 'No'
        return (f'QENS Result for Dataset {self.name}\n'
                f'Fit converged: {converged}\n'
                f'ier = {self.ier}, : {self.mesg}'
                f'\nchi^2 = {self.chisq}\n')

    def validate_result(self):
        """
        Checks if all the arguments have been passed properly to the Result
        instance after the fit has been run.

        Raises
        ------
        RuntimeError
            If any argument for QENSResult is None, an error is raised.

        Returns
        -------
        None.

        """
        if all(i is None for i in [self.x,
                                   self.y,
                                   self.params,
                                   self.chisq,
                                   self.popt,
                                   self.pcov,
                                   self.infodict,
                                   self.mesg,
                                   self.ier,
                                   self.residuals]):
            raise RuntimeError('Result incomplete, an argument was not passed')

    def print_result(self, index = None, index_title = None):
        """
        Prints all the best fit parameters in a nice readable way, as a
        pandas.DataFrame. The index and its title can be customised.

        Parameters
        ----------
        index : TYPE, optional
            Index values for the DataFrame. The default is None.
        index_title : TYPE, optional
            Index title for the DataFrame. The default is None.

        Returns
        -------
        None.

        """
        if self.ier in [1, 2, 3, 4]:
            converged = 'Yes'
        else:
            converged = 'No'
        print(f'Dataset: {self.name}\nFit converged: {converged}.  '
              f'ier = {self.ier}. Message: {self.mesg} chi^2 = {self.chisq}')
        print('Parameter values:')
        print(self.params.all_to_df(index, index_title), '\n\n')

class Model:
    def __init__(self,
                 func: callable,
                 parlist: list,
                 datasets: dict = None,
                 **kwargs):
        self.target = func
        self.pnames = []
        self.cnames = []
        if datasets is None:
            raise RuntimeError('No dataset provided!')
        self.ds = datasets
        self.constants = kwargs
        self.res = dict.fromkeys(self.ds.keys())
        self.n_ds = len(self.ds)
        self._check_params(parlist)
        self._par_cycler = None

    def __repr__(self):
        return (f'Model object tied to function {self.target}\n\n'
                f'Parameters:\n{self.params}\n\n'
                f'Data: {self.ds}\n')

    def _check_params(self, parlist: list):
        self.sig = inspect.signature(self.target)

        lst = []
        pardict = {}
        for i in parlist:
            pardict[i.name] = i

        for pname, par in self.sig.parameters.items():
            if par.kind == par.POSITIONAL_OR_KEYWORD:
                self.pnames.append(pname)
                if pname not in pardict:
                    raise RuntimeError(f'Parameter {pname} was not ' +
                                       'declared in the parameter list.')
                lst.append(pardict[pname])
            elif par.kind == par.KEYWORD_ONLY:
                self.cnames.append(pname)
                if pname not in list(self.constants) + ['q']:
                    raise RuntimeError(f'Constant {pname} was not ' +
                                       'passed to the Model object.')
        self.params = ParList(lst, list(self.ds.values())[0].n_q)

    def _helper(self, data, x, *par):
        x = x.reshape(data.n_q, -1)
        result = np.zeros((data.n_q, len(x[0])))
        self.params.pack_values(par)
        for i in range(data.n_q):
            params = tuple([self.params[key][i] for key in self.pnames])
            constants = {'q': data.q[i]}
            for cname, cvalue in self.constants.items():
                constants[cname] = (cvalue[i] if isinstance(cvalue, np.ndarray)
                                    else cvalue)
            result[i] = self.target(x[i], *params, **constants)

        return result.flatten()

    def run_fit(self):
        with alive_bar(self.n_ds) as bar:
            for key in self.ds:
                self.ds[key].name = key
                self.res[key] = QENSResult(name = self.ds[key].name)

                (self.res[key].popt,
                self.res[key].pcov,
                self.res[key].infodict,
                self.res[key].mesg,
                self.res[key].ier) = curve_fit(
                    partial(self._helper, self.ds[key]),
                    self.ds[key].x.flatten(),
                    self.ds[key].y.flatten(),
                    sigma = self.ds[key].dy.flatten(),
                    p0 = self.params.unpack_pin(),
                    bounds = self.params.unpack_bounds(),
                    nan_policy='omit',
                    full_output = True,
                    absolute_sigma = True)

                norm_resid = self.res[key].infodict['fvec']
                self.res[key].chisq = sum(
                    j*j for j in norm_resid) / len(norm_resid)
                self.res[key].residuals = norm_resid.reshape(
                    self.ds[key].n_q, -1)
                self.res[key].x = np.array([np.linspace(arr.min(),
                                              arr.max(),
                                              500) for arr in self.ds[key].x])
                self.res[key].y = self._helper(
                    self.ds[key],
                    self.res[key].x,
                    *self.res[key].popt).reshape(self.ds[key].n_q, -1)

                self.params.pack_values(self.res[key].popt)
                self.params.pack_errors(3*np.sqrt(np.diag(self.res[key].pcov)))
                self.res[key].params = copy.deepcopy(self.params)
                self.res[key].validate_result()
                self.res[key].print_result(index = self.ds[key].q,
                                       index_title = 'q')
                bar()

    def plot_fits(self,
                  data_only = False,
                  xlabel: str = 'E (meV)',
                  ylabel: str = 'Scattering Intensity (A.U.)',
                  **plt_kw):
        for key in self.ds:
            fig, ax = plt.subplots(
                2 if not data_only else 1 ,
                self.ds[key].n_q,
                sharey = 'row',
                sharex = 'col',
                figsize  = (16,9),
                layout = 'tight',
                height_ratios = [4,1] if not data_only else None,
                squeeze = False)
            for i in range(self.ds[key].n_q):
                ax[0,i].errorbar(self.ds[key].x[i],
                                 self.ds[key].y[i],
                                 self.ds[key].dy[i],
                                 **plt_kw)
                ax[0,i].yaxis.set_tick_params(labelleft=True)
                ax[0,i].xaxis.set_tick_params(labelbottom=True)
                ax[0,i].set_xlabel(xlabel)
                ax[0,i].set_ylabel(ylabel)

                if not data_only:
                    ax[0,i].plot(self.res[key].x[i],
                                 self.res[key].y[i])

                    ax[1,i].plot(self.ds[key].x[i],
                                 self.res[key].residuals[i])
                    ax[1,i].yaxis.set_tick_params(labelleft=True)
                    ax[1,i].xaxis.set_tick_params(labelbottom=True)
                    ax[1,i].set_xlabel(xlabel)
                    ax[1,i].set_ylabel('Normalised Residuals')
                    ax[1,i].set_ylim(bottom = -5, top = 5)
                    ax[1,i].axhline(y=0, color = 'g', linestyle = '--')
                    ax[1,i].axhline(y=-3, color = 'r', linestyle = '--')
                    ax[1,i].axhline(y=3, color = 'r', linestyle = '--')



            self.res[key].cycler = SubplotCycler(
                fig, ax) if self.ds[key].n_q > 1 else None

    def plot_par(self,
                 index = 'q',
                 x_title = r'$q\ (\AA^{-1})$',
                 x_title_glob = r'$T\ (K)$',
                 **pltpar_kw):
        glob_data = pd.DataFrame()
        fig_par, ax_par = plt.subplots(
            1,
            self.n_ds * self.params.n_free,
            figsize  = (6 * self.params.n_free,6),
            layout = 'tight',
            squeeze = False)
        for j, key in enumerate(self.res):
            if isinstance(index, str):
                if index == 'q':
                    index = self.ds[key].q
                else:
                    index = self.ds[key].constants[index]
            plt_data = self.res[key].params.free_to_df(
                index,
                index_title = x_title)
            nf = self.res[key].params.n_free
            for i in range(nf):
                ax_par[0,j*nf + i].errorbar(plt_data.index,
                                     plt_data.iloc[:,2*i],
                                     plt_data.iloc[:,2*i+1],
                                     **pltpar_kw)
                ax_par[0,j*nf + i].set_ylabel(plt_data.columns[2*i])
                ax_par[0,j*nf + i].set_xlabel(plt_data.index.name)

            glob_data = pd.concat(
                [glob_data, self.res[key].params.global_to_df(
                    index = [key], index_title = x_title_glob)])
        self._par_cycler = SubplotCycler(
            fig_par,
            ax_par,
            simultaneous_plots = nf) if self.n_ds > 1 else None
        if self.n_ds > 1:
            fig_glob, ax_glob = plt.subplots(
                1,
                self.params.n_global,
                figsize  = (6 * self.params.n_global,6),
                layout = 'tight',
                squeeze = False)
            for i in range(self.params.n_global):
                ax_glob[0,i].errorbar(glob_data.index,
                                     glob_data.iloc[:,2*i],
                                     glob_data.iloc[:,2*i+1],
                                     **pltpar_kw)
                ax_glob[0,i].set_ylabel(glob_data.columns[2*i])
                ax_glob[0,i].set_xlabel(glob_data.index.name)

def load_ascii(filename: str,
              sep: str = ',',
              index_list: list = []):

    print(f'Current Path: {os.getcwd()}')

    names = glob.glob(filename)
    if not names:
        raise RuntimeError('No file found, check the name or the path.')

    print(f'Opening files: {names}')

    name_dict = {}
    out_dict = {}

    if index_list is None:
        index_list = [i for i in range(len(names))]
        for i in index_list:
            name_dict[i] = names[i]
    elif index_list != []:
        if len(index_list) != len(names):
            raise RuntimeError('Number of files found and length of index_list'
                               " don't match!")
        else:
            for i, index in enumerate(index_list):
                name_dict[index] = names[i]
    else:
        for i, name in enumerate(names):
            temp = re.findall(r'_(\d+\.\d+)K', name)
            if (len(temp) > 1) or (temp == []):
                raise RuntimeError('Impossible to infer the temperature '
                                   'from the file name. Please rename it to '
                                   'filename_xxx.xxxK.csv or provide your own '
                                   'index list.')
            name_dict[float(temp[0])] = name

    for key, file in name_dict.items():
        with open(file) as f:
            q, x, y, dy = [], [], [], []
            next(f)
            for line in f:
                values = line.split(sep)
                if len(values) > 1:
                    x.append(float(values[0]))
                    try:
                        y.append(float(values[1]))
                    except:
                        y.append(float('nan'))
                    try:
                        dy.append(float(values[2]))
                    except:
                        dy.append(float('nan'))
                else:
                    q.append(float(values[0]))

        out_dict[key] = QENSDataset(name = key,
                                    x = np.array(x).reshape((len(q), -1)),
                                    y = np.array(y).reshape((len(q), -1)),
                                    dy = np.array(dy).reshape((len(q), -1)),
                                    q = np.array(q))

    return out_dict

if __name__ == "__main__":

    plt.close('all')

    def fitmodel(x, /, A, omega, phi, *, q):
        return A * np.sin(omega * x + phi) * np.exp(-x / q)

    lst = [Parameter('omega', [0.2, 0.6, 1.7], 0., 50.),
           Parameter('phi', 1., 0., 5., is_global = True),
           Parameter('A', 10., 5., 25., is_fixed = False),]

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

    data = {'1': QENSDataset(x = x, y = y, dy = dy, q = q),
            '5': QENSDataset(x = x, y = y2, dy = dy, q = q)}

    mod = Model(fitmodel, lst, data)
    mod.run_fit()
    mod.plot_fits(fmt = ' o')
    mod.plot_par(fmt = ' o')
