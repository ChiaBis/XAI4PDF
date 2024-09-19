# This fork 

The scope of this repository, which forks the public `XAI4PDF` repo created by Brandon Kriesten, Jonathan Gomprecht, and T.J. Hobbs, is to try and apply explainable AI to another set of PDFs. 
Specifically, the analysis will be done with PDFs coming from a NN fit -- `NavyPier PDFs`.







# Original repo README.md
README: XAI4PDF, July 2024.

The XAI4PDF project is a set of Python analysis modules which produce
the results of Explainable AI Classification for Parton Density Theory; 
the code is developed by the authors of that study Brandon Kriesten, 
Jonathan Gomprecht, and T.J. Hobbs. Questions should be referred 
to bkriesten@anl.gov and/or tim@anl.gov.

The current package includes the ResNet-like model architecture in
resnet_model.py. There are two Jupyter notebooks, one of which named
flavor_classifier_analysis.ipynb which contains the results of XAI4PDF
for flavor classification as discussed in the manuscript. The second
Jupyter notebook named pheno_classifier_analysis.ipynb contains XAI4PDF
for classification of phenomenological fits. Given the large file sizes,
we have split the training data into two files pheno_class_x_test_1.npy
and pheno_class_x_test_2.npy, as well as pheno_class_y_test.npy. We also
include the pdf data array for the flavor classifier which is called
pdf_arr_interpret_data.npy. For convenience we provide the standard scaler
models in the form of pickle files flavor_scaler.pkl and pheno_class_scaler.pkl.
Finally we have also included fully trained model weights which can be
loaded into the analysis Jupyter notebooks called rn.chkpnt.weights.h5 for
the pheno classifier and pdf_ratio_classifier.hdf5 for the parton
flavor classifier.

