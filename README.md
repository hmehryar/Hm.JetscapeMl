# Hm.JetscapeMl

Welcome to Hm.JetscapeMl! This repository contains code and resources related to utilizing machine learning techniques for analyzing Jetscape simulation data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Hm.JetscapeMl is a project aimed at applying machine learning methods to analyze data from Jetscape simulations, a toolkit used for simulating the quark-gluon plasma created in heavy-ion collisions. The goal is to extract valuable insights and patterns from the simulation data using modern machine learning techniques.

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

