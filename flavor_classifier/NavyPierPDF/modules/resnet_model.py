# Importing necessary libraries for numerical computations, data manipulation,
# and visualization
import numpy as np               # Numerical computing with arrays
import pandas as pd              # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting and visualizations
# import lhapdf                  # (Commented out) Could be used for PDF data in physics but not currently active

# Importing TensorFlow and Keras for deep learning and model building
import tensorflow as tf
from tensorflow.keras import models, optimizers, layers, losses, regularizers, callbacks

# Importing scikit-learn modules for data preprocessing and model evaluation
from sklearn.model_selection import train_test_split    # Splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler        # Standardizing features by removing the mean and scaling to unit variance
from sklearn.preprocessing import OneHotEncoder         # Encoding categorical features into one-hot vectors

# Additional matplotlib modules for more advanced plotting features
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Allows creation of adjustable colorbars in plots
import matplotlib.colors as mcolors                      # For color mapping and normalization in visualizations
# import pickle                                          # (Commented out) Could be used for saving/loading models or data, but not currently in use



def build_xaxis():
    """
    Builds a composite array of x-axis values for plotting or data generation.

    The x-axis consists of two parts:
    1. A logarithmically spaced set of values ranging from 10^-4 to 10^-1.
    2. A linearly spaced set of values ranging from 0.1 to just below 0.4.

    Returns:
        xs (numpy array): The combined array of logarithmic and linear x-axis values.
    """

    # Generate 100 logarithmically spaced values between 10^-4 and 10^-1
    low_xs = np.array([x for x in np.logspace(-4, -1, 100)])

    # Generate 100 linearly spaced values between 0.1 and 0.4 (exclusive)
    high_xs = np.array(np.arange(0.1, 0.399, 0.3 / 100.))

    # Concatenate the logarithmic and linear arrays to create a composite x-axis
    xs = np.hstack([low_xs, high_xs])

    return xs


def create_pdf_data():
    """
    Loads PDF (Parton Distribution Function) data and generates labeled PDF arrays for further analysis.
    
    NOTE: This function loeds an external file `pdf_arr_interpret_data.npy`, which must be stored in
    the `inputs` directory, i.e., in the path `/inputs/` from the working directory (main jupyter nb).
    
    The function:
    1. Calls the `build_xaxis` function to generate the x-axis values.
    2. Loads a pre-processed NumPy array (`pdf_arr_interpret_data.npy`) containing PDF data.
    3. Creates an array of labeled strings for the PDF components (gluons, quarks, and antiquarks).

    Returns:
        pdf_arr (numpy array): The loaded array containing the PDF data.
        pdf_labels (numpy array): An array of LaTeX-styled labels for the PDF data components.
    """

    # Load a NumPy array containing the PDF data from a .npy file
    # This file should contain pre-processed data for various partons (gluons, quarks, antiquarks, etc.)
    pdf_arr = np.load("/inputs/pdf_arr_interpret_data.npy")

    # Create an array of LaTeX-styled labels for the different PDF components
    pdf_labels = np.array(
        [
            r"$g/g_{0}$",  # Gluon ratio
            r"$s/s_{0}$",  # Strange quark ratio
            r"$u/u_{0}$",  # Up quark ratio
            r"$d/d_{0}$",  # Down quark ratio
            r"$\bar{u}/\bar{u}_{0}$",  # Anti-up quark ratio
            r"$\bar{d}/\bar{d}_{0}$",  # Anti-down quark ratio
        ]
    )

    # Return the loaded PDF data and corresponding labels
    return pdf_arr, pdf_labels


def plot_pdf_data():

    xs = build_xaxis()
    pdf_arr, pdf_labels = create_pdf_data()
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,25))

    for i in range(len(pdf_arr)):
    
        axMain = plt.subplot(3,2,i+1)
        axMain.plot(xs,pdf_arr[i,:250].T)
        axMain.set_xscale('linear')
        axMain.set_xlim((0.101, 0.4))
        axMain.set_xticks([0.2,0.3,0.4])
        axMain.spines['left'].set_visible(False)
        axMain.yaxis.set_ticks_position('right')
        axMain.yaxis.set_visible(False)
        plt.xticks(fontsize=25)


        divider = make_axes_locatable(axMain)
        axLin = divider.append_axes("left", size=2.0, pad=0, sharey=axMain)
        axLin.set_xscale('log')
        axLin.set_xlim((1e-4, 0.1))
        axLin.plot(xs,pdf_arr[i,:250].T)
        axLin.spines['right'].set_visible(False)
        axLin.yaxis.set_ticks_position('left')
        plt.setp(axLin.get_xticklabels(), visible=True)

        if i == 1:
            plt.ylim(0.1,1.9)
        else:
            plt.ylim(0.75,1.25)
        plt.ylabel(pdf_labels[i],fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        axMain.tick_params(axis='x', which='major', pad=18)
        axLin.tick_params(axis='x', which='major', pad=18)
        plt.tick_params(axis='y', which='major', pad=2)

        axMain.annotate(r'$x$', xy=(0.2, -0.13), xycoords='axes fraction', fontsize=35, ha='center', va='center')

    plt.savefig('pdf_data.pdf', format='pdf')

    plt.show()


class PDFClassifier(models.Model):
    
    def __init__(self, input_dim, num_classes, regularization=1e-4):
        super(PDFClassifier,self).__init__()
        
        self.input_size = (input_dim,)
        self.output_size = num_classes
        self.reg = regularization
        
        self.model = self.build_classifier()

    def build_classifier(self):

        #Model input
        inputs = tf.keras.Input(shape=self.input_size,
                                name='input_layer')

        ### Dense unit 1
        x1 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_1')(inputs)
        x2 = layers.Activation(activation='relu', 
                               name='activation_1')(x1)
        x2 = layers.Dropout(rate=0.5,
                            name='dropout_1')(x2)
        
        ### Dense unit 2
        x2 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_2')(x2)
        x2 = layers.Activation(activation='relu',
                               name='activation_2')(x2)
        x2 = layers.Dropout(rate=0.5,
                            name='dropout_2')(x2)
        
        ### Dense unit 3 w/ resnet connection 1
        x3 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_3')(x2)
        sc1 = layers.Add(name='shortcut_1')([x1, x3])
        x3 = layers.Activation(activation='relu',
                               name='activation_3')(sc1)
        x3 = layers.Dropout(rate=0.5,
                            name='dropout_3')(x3)

        ### Dense unit 4 w/ resnet connection 2
        x4 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_4')(x3)
        sc2 = layers.Add(name='shortcut_2')([x2, x4])
        x4 = layers.Activation(activation='relu',
                               name='activation_4')(sc2)
        x4 = layers.Dropout(rate=0.5,
                            name='dropout_4')(x4)

        ### Dense unit 5 w/ resnet connection 3
        x5 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_5')(x4)
        sc3 = layers.Add(name='shortcut_3')([x3, x5])
        x5 = layers.Activation(activation='relu',
                               name='activation_5')(sc3)
        x5 = layers.Dropout(rate=0.5,
                            name='dropout_5')(x5)

        ### Dense unit 6 w/ resnet connection 4
        x6 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_6')(x5)
        sc4 = layers.Add(name='shortcut_4')([x4, x6])
        x6 = layers.Activation(activation='relu',
                               name='activation_6')(sc4)
        x6 = layers.Dropout(rate=0.5,
                            name='dropout_6')(x6)

        ### Dense unit 7 w/ resnet connection 5
        x7 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_7')(x6)
        sc5 = layers.Add(name='shortcut_5')([x5, x7])
        x7 = layers.Activation(activation='relu',
                               name='activation_7')(sc5)
        x7 = layers.Dropout(rate=0.5,
                            name='dropout_7')(x7)

        ### Dense unit 8 w/ resnet connection 6
        x8 = layers.Dense(units=1000, 
                          kernel_regularizer=regularizers.l2(self.reg),
                          name='dense_8')(x7)
        sc6 = layers.Add(name='shortcut_6')([x6, x8])
        x8 = layers.Activation(activation='relu',
                               name='activation_8')(sc6)
        x8 = layers.Dropout(rate=0.5,
                            name='dropout_8')(x8)

        ### Model Outputs with softmax
        x9 = layers.Dense(units=self.output_size,
                          name='dense_output')(x8)
        outputs = layers.Activation(activation='softmax',
                                    name='activation_softmax')(x9)

        ### Construct the model from inputs/outputs
        model = models.Model(inputs=inputs,
                             outputs=outputs,
                             name='classifier_model')
        return model

    def call(self,x):
        logits = self.model(x)
        return logits

def split_data():
    xs = build_xaxis()
    pdf_arr, pdf_labels = create_pdf_data()
    pdf_data = pdf_arr.reshape(pdf_arr.shape[0]*pdf_arr.shape[1], pdf_arr.shape[2])
    label_data = np.vstack([np.full(shape=(10_000,1),fill_value=0), np.full(shape=(10_000,1),fill_value=1),
                            np.full(shape=(10_000,1),fill_value=2), np.full(shape=(10_000,1),fill_value=3),
                            np.full(shape=(10_000,1),fill_value=4), np.full(shape=(10_000,1),fill_value=5)
                           ])
    x_train,x_test,y_train,y_test = train_test_split(
        pdf_data, label_data, stratify=label_data, shuffle=True, random_state=42, test_size=0.3
    )
    x_valid,x_test,y_valid,y_test = train_test_split(
        x_test, y_test, stratify = y_test, shuffle=True, random_state=42, test_size=0.5
    )

    label_encoder = OneHotEncoder()
    sc = StandardScaler()

    y_train = label_encoder.fit_transform(y_train.reshape(-1,1)).toarray()
    y_valid = label_encoder.fit_transform(y_valid.reshape(-1,1)).toarray()
    y_test = label_encoder.transform(y_test.reshape(-1,1)).toarray()

    x_train = sc.fit_transform(x_train)
    x_valid = sc.transform(x_valid)
    x_test = sc.transform(x_test)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def train_model():
    
    pdf_arr, pdf_labels = create_pdf_data()
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data()
    
    num_classes = pdf_arr.shape[0]
    class_model = PDFClassifier(num_classes)
    class_model.compile(optimizer=optimizers.legacy.Adam(learning_rate=1e-4),
                        loss = losses.CategoricalCrossentropy())
    class_model.build(x_train.shape)

    early_stopping_callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=False,
        mode='auto'
    )

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath='text.hdf5',
        monitor='val_loss',
        verbose=True,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_frequency=1
    )

    history = class_model.fit(
        x = x_train,
        y = y_train,
        batch_size=32,
        shuffle=True,
        epochs=1000,
        validation_data = [x_valid,y_valid],
        callbacks=[early_stopping_callback,
                   model_checkpoint_callback]       
    )
    
    return history
