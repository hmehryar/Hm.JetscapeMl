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

