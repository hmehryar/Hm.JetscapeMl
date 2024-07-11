# Hm.JetscapeMl

Welcome to Hm.JetscapeMl! This repository contains code and resources related to utilizing machine learning techniques for analyzing Jetscape simulation data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Hm.JetscapeMl is designed to extract valuable insights and patterns from Jetscape simulation data using modern machine learning techniques. The dataset and accompanying scripts provide a comprehensive framework for conducting machine learning experiments on this data.

## Dataset
The dataset is hosted on Kaggle: [ML-Jet Dataset](https://www.kaggle.com/datasets/haydarmehryar/ml-jet) ([https://www.kaggle.com/datasets/haydarmehryar/ml-jet](https://www.kaggle.com/datasets/haydarmehryar/ml-jet)).



## Installation

To get started with Hm.JetscapeMl, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/hmehryar/Hm.JetscapeMl.git
```

2. Navigate to the repository's directory:

```bash
cd Hm.JetscapeMl
```

3. Reading the Dataset: To read and utilize the dataset, users can employ various tools and libraries compatible with the pickle format. Here’s how the dataset can be accessed using Python with the `pickle` library:

```bash
import pickle
dataset_file_name = f"ml_jet_dataset.pkl"
try:
    with open(file_name, 'rb') as dataset_file:
        loaded_data = pickle.load(dataset_file, encoding='latin1')
        (dataset_x, dataset_y) = loaded_data
        print("dataset_x:",type(dataset_x), dataset_x.size, dataset_x.shape)
        print("dataset_y:",type(dataset_y), dataset_y.size,dataset_y.shape)
except pickle.UnpicklingError as e:
        print("Error while loading the pickle file:", e)
```
# Repository Guideline
## Rebuidling/Expanding Dataset
All the step-by-step process/related codes for buidling the ML-JET Dataset can be found in [jet_ml_dataset_builder](https://github.com/hmehryar/Hm.JetscapeMl/tree/main/jet_ml_dataset_builder) Directory. 

## Applying Machine Learning (ML) & Neural Network (NN) Architectures
### Neural Networks
#### MNIST Net
**MNIST Net** ~\cite{lecun1998gradient}, more commonly known as *LeNet*. It was initially devised for handwritten digit recognition, leverages insights into 2D shape invariances through local connection patterns and weight constraints. It uses an image input. With 4 layers, including convolutional and fully connected layers, MNIST Net boasts 96,445 trainable parameters. The model implementation can be found at [mnist]() Direcetory.

#### VGG16 Net
**VGG16Net** ~\cite{simonyan2014very}, renowned for its remarkable performance in image recognition tasks. It uses an image input and comprises 16 layers, with 4 convolutional and fully connected blocks, totaling 15,676,673 trainable parameters. The model implementation can be found at [vgg16]() Direcetory.

#### Point Net
**PointNet**~\cite{qi2017pointnet} introduces a novel approach to processing point cloud data, making it uniquely suited for our jet event image classification task. Unlike conventional CNNs that operate on structured grid-like data, PointNet directly consumes unordered point sets. The model implementation can be found at [pointnet]() Direcetory.
### Traditional Machine Learning
#### Logistic Regression
- The logistic regression model is trained specifically for binary classification on the first column.
- Predictions and evaluation are performed based on the binary labels.

#### Decision Tree
This code uses DecisionTreeClassifier instead of LogisticRegression. The structure is similar: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

#### Support Vector Machine (SVM)
This code uses LinearSVC instead of LogisticRegression or DecisionTreeClassifier. The structure remains similar: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

#### K-Nearest Neighbors (KNN)
Adjust the k_neighbors parameter based on your requirements. The structure is similar to the previous examples: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy.

#### Random Forest
This code uses RandomForestClassifier from scikit-learn. The structure is similar to the previous examples: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

 #### [jet_ml_dataset_builder_by_size](jet_ml_dataset_builder_by_size) For building a new dataset from the original dataset with different

 #### [jet_ml_diffusion_model](jet_ml_diffusion_model) *Difussion model* implemeteation for generating events from parameters

 #### [jet_ml_mnist_net](jet_ml_mnist_net) *MNIST Net* Implemetation of binary classifier for Eloss for each 9 different configuration

 #### [jet_ml_models](jet_ml_models) consists of python classes for each implmented models (For now just pointnet has its own implementation)

 #### [jet_ml_synthesis_model_vgg16](jet_ml_synthesis_model_vgg16) consists of *VGG16 Net* implemetation, user can choose the desired parameter from $eloss$, $\alpha_s$, or $Q_0$ to train the classifier for it
 #### [jet_ml_validation_calculator](jet_ml_validation_calculator) consists of implmentation for loading a trained model and calculate the confusion matrix and accuracy for it based on the loaded dataset

 #### [jet_ml_vgg16_model_cnn](jet_ml_vgg16_model_cnn) *VGG16 Net* Implemetation of binary classifier for Eloss for each 9 different configuration
 #### [jet_ml_models_notebooks](jet_ml_models_notebooks) it consists the binary classification implemetation of *Decision Tree*, *KNN*, *Random Forest*, *SVM*,and *Logistic Regression* for $eloss$, more detail explanation of code implemented in this directory is in Traditional machine learning method section.
 #### [jet_ml_pointnet](jet_ml_pointnet) several implmentation of PointNet for binary classification of $eloss$ or a single notebook that can train different type of classifier based on user desire, and it includes a sample tensorflow GPU implmentation 
 #### [jet_ml_pointnet_alpha_s](jet_ml_pointnet_alpha_s) includes a PointNet classifier specifically for $\alpha_s$
 #### [jet_ml_pointnet_eloss](jet_ml_pointnet_eloss) includes a PointNet binary classifier specifically for $eloss$
 #### []()
 #### []()
 #### []()

## Usage
Once you have the repository set up and the dependencies installed, you can start utilizing the project:
1. Data Preprocessing: Use the provided scripts to preprocess and prepare Jetscape simulation data for analysis.
2. Machine Learning Models: Explore the models directory to find pre-implemented machine learning models tailored for analyzing Jetscape data.
3. Example Notebooks: Check out the notebooks directory for example Jupyter notebooks that demonstrate how to use the machine learning models with Jetscape data.
4. Customization: Feel free to customize and extend the code to suit your specific needs and experiments.

   
## Contributing
Contributions are welcome and encouraged! If you'd like to contribute to Hm.JetscapeMl, follow these steps:

1. Fork the repository to your GitHub account.
2. Create a new branch from the main branch for your changes.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Open a pull request (PR) to the original repository, describing the changes you've made.
   
Please ensure your contributions adhere to the project's coding standards and follow best practices.

## License
This project is licensed under the [MIT License](https://github.com/hmehryar/Hm.JetscapeMl/blob/main/LICENSE).
Feel free to customize this template according to your project's specifics and additional information you want to provide.

