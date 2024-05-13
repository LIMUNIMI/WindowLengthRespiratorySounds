# WindowLengthRespiratorySounds

__This is the experiment of my thesis. The main purpose is to see analyze how the parameter 'window length' affects
the performance of an AutoML system.__

Link to dataset: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

1- Download the dataset and organize the files, dividing wav and txt files in separate directories.
2- Use the first functions in the functions.py script to operate segmentation (split each audio files in segments
wich represent the respiratory cycles).
3- Extract audio features from segments using the chosen windowlength value.
4- Split feature dataframe in train set and test set, using indications provided.
5- Produce Healthy/Unhealthy labels and balance the train set.
6- Generate the ensemble using Autosklearn 2.0 and save them as binary files using pickle
7- Refit operation to skip cross validation
8- Prediction and evaluation metrics

Use tirocinio.yml to import the conda enviroment with all the python libraries used during the experiment.
