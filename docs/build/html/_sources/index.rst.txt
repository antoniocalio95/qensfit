QENSFit documentation
=====================

**QENSFit** is a little library that enables easy fitting of
Quasielastic Neutron Scattering data.

It borrows some concepts from the popular lmfit package,
such as the Model and Parameter objects, but everything is
implemented from scratch to make it compatible with
Global Fitting procedures, which are essential for QENS data.

USAGE:

* Use LoadAscii to load data reduced with Mantid, or load your
  data another way and then organise it in a dictionary of
  QENSDataset instances. Loading of multiple datasets is supported.

* Define your model function to fit the data. Constants are also
  supported.

* Declare a list of Parameter obejcts to control their initial
  value, bounds, and whether the parameter is fixed and/or global.

* Declare a model instance using the function you wrote before
  as the target, and feed it the list of Parameter objects and
  the QENSDataset.

* Run the fit.

* Plot fits and parameters, and Save your results.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Home Page <self>
   Examples <examples>
   API Reference <api>