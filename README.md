# LearningToDrive
This project is based on [ICCV 2019: Learning-to-Drive Challenge](https://www.aicrowd.com/challenges/iccv-2019-learning-to-drive-challenge). This project wins #3 in the competition.  
## Competition description
### Goal
The goal of this challenge is to develop state of the art driving models that can predict the future steering wheel angle and vehicle speed given a large set of sensor inputs.  
### Sensor data
A [video](https://youtu.be/mnnSf2KwTS4) visualing the sensor data. 
We supply the Drive360 dataset consisting of the following sensors:
* 4xGoPro Hero5 cameras (front, rear, right, left)
* Visual map from HERE Technologies
* Visual map from TomTom (may not be perfectly synced)
* Semantic map from HERE Technologies

The Drive360 dataset is split into a train, validation and test partition.
### Task
Challenge participants are tasked to design, develop and train a driving model that is capable of predicting the steering wheel angle and vehicle speed obtained from the vehicles CAN bus 1 second in the future.  
Challenge participants can use any combination of camera images, visual map images and semantic map information as input to their models. It is also allowed to use past sensor information in addition to the present observations, however it is NOT allowed to use future sensor information.
### Evaluation
Driving Models will be evaluated on the test partition using the mean-squared-error (MSE) performance for both steering wheel angle (‘canSteering’) and vehicle speed (‘canSpeed’) predictions with the human ground truth as a metric. Thus best performing driving models will drive identical to the human driver in these situations.

## How to build and run
1. Run `pip install -r requirements.txt` to install the necessary packages.
2. Download the complete dataset (150G) from [here](https://www.aicrowd.com/challenges/iccv-2019-learning-to-drive-challenge/dataset_files) or the properly downsampled dataset (700M) from [here](https://drive.google.com/file/d/1fc5vFlnpWjxuAqQxqqJKU26DjFM7s6fo/view?usp=sharing)
3. Run LearningToDrive.ipynb using Jupyter Notebook. 
