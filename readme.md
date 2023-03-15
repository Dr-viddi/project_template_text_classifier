## About this project
In this project, a text classifier is implemented. That is for a provided text string, a text suggestion will be predicted. For this supervised learning task, 6 models are implemented: support vector machine, naive bayes, logistic regression, random forest, LSTM, CNN. The code ist containerized and the prediction is done via a web app.


## Quickstart
1. Create new environment with
```
conda create --name myenv python=3.9
```

2. Activate the environment with 
```
conda activate myenv
```

3. Navigate to root folder *text_classifier/* and enter:
```
pip install .
```

4. Download the trainingsdata from
```
...
```
and add it in the */data* folders 


5. Train a preconfigurated support vector machine model via
```
start_training_pipeline
```
which only takes the 4 most represented text classes into account to tackle the imbalanced dataset. In a previous analysis, the support vector machine model was the best performing model. The output files (plots, trained models and vectorizer, classification reports) are created and placed in the *runs* folder.

6. Dockerize and start the web app via
```
docker-compose up
```

7. Access the web app and do a prediction via your browser by entering in the address bar:
```
127.0.0.1:8000/suggest?input=<user_string>
```
where *<user_string>* is the text for which a prediction should be made.


## Prerequisite
Anaconda, python and pip must be installed on the machine.


## Installation
A new environment can be created via
```
conda create --name myenv python=3.9
```
and confirm with *y*. Next, activate the environment via
```
conda activate myenv
```
Navigate to the root folder and enter
```
pip install .
```
to build the text_classifier package. Note that only the most necessary packages are defined to keep the environment simple and clean. Pip may display dependency errors for packages that are not needed. However, this will not affect the functionality of the programm.


## Training
You can either download pretrained models or train a model locally. 

### 1. Download pretrained models
You can download pretrained models from the following link:
```
...
```
Just copy the *labels.pickle*, *model.pickle*, and *tokenizer.pickle* files in the *runs* folder and the three config files in the *configs* folders. A tsupport vector machine model and a LSTM model, each trained on the dataset containing all classes and containing only the 4 most common classes, are provided on the drive for convenience. The support vector machine was the best overall performing model and the LSTM model was the best performing deep learning model.


### 2. Train a model locally

#### Download trainings data:
Training data can be downloaded 
```
https://drive.google.com/file/d/1es3EX0MdDAeolwFl_K_fS3RP0JFRxE2U/view?usp=sharing
```
and has to be placed in "data" folder.

#### Setup Configurations
In the *configs* folder, different types of config files are provided:
* pipeline_config: The overall config file for the training pipeline containing storage paths, preprocessings, model type and training parameters.
* model_config: Containing the parameters for the different models, that is specified in the *pipeline_config* file.
For each implemented model (support vector machine, naive bayes, logistic regression, random forest, LSTM, CNN), a *pipeline_config* and  a *model_config* is provided, that can be directly used or changed.


#### Training
After the configuration files are set, training a model can be done by just entering the command
```
start_training_pipeline
```
Per default, the trainings pipeline is executed that is defined in the default config file *pipeline_config.yml* in the config folder. You can also provide training_config files via the flag *-c*. For example:
```
start_training_pipeline -c configs/my_config.yml
```
executes the training pipeline with the configuration provided in the my_config.yml file. Here, for each model, a default configuration is provided that can be executed via:
```
start_training_pipeline -c configs/pipeline_config_svm.yml
start_training_pipeline -c configs/pipeline_config_naive_bayes.yml
start_training_pipeline -c configs/pipeline_config_random_forest.yml
start_training_pipeline -c configs/pipeline_config_log_reg.yml
start_training_pipeline -c configs/pipeline_config_lstm.yml
start_training_pipeline -c configs/pipeline_config_cnn.yml
```
After training, several files are created:
* a model file in the runs folder storing the trained model
* a trained tockenizer in the runs folder storing the fitted tokenizer
* a label file in the runs folder containing the used labels
* a figure fig_class_count showing the classes of the training data
* a figure fig_accuracy showing the training accuracy (only when LSTM or CNN model is selected)
* a figure fig_loss showing the training loss (only when LSTM or CNN model is selected)
* a figure fig_confusion_matrix showing the confusion matrix
* a text file classification_report showing the values of the metrics.
Please note, after the training a *pipeline_config* file is created containing the information from training. Only the trained model that is specified in the *pipeline_config* is used for the prediction.


## Run and predict
After training, a web app for the prediction can be created and started via
```
docker-compose up
```
If the command is executed the first time, the docker container will be build which may take few minutes. Afterwards, the container will be startet. The app can then be accessed and a prediction can be made via an inter browser via:
```
127.0.0.1:8000/suggest?input=<user_string>
```
where *<user_string>* is a string that should be classified. Alternatively, one can use fastAPI's swagger UI
```
http://127.0.0.1:8000/docs
```
and navigate to the *suggest* call. Note that if a another model is trained/used for prediction (or the codebasis is changed) a new docker build has to be trigger via
```
docker-compose build
```
to update the docker container. A new execution of the command *docker-compose up* will then start the app with the new model. With 
```
http://127.0.0.1:8000/info
```
the currently deployed model is shown.


## Code quality

### Run tests
In root, just use the command
```
pytest
```
to run all unittests.

### Pre-Commit
*pre-commit* can be activated by running
```
pre-commit install
```
and the precomits can be manually executed via
```
pre-commit run all
```
Note that no git repository is is initialized and thus, pre-commit will fail. If *pre-commit* will be used anyways, a dummy repo can be created via
```
git init
```