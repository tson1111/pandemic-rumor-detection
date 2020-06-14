## Test Environment
Ubuntu 18.04 LTS

## Requirement 
- PaddlePaddle 1.8
- Fool NLTK
  - Tensorflow <= 1.14.0, otherwise you have to manually modify some method names called by Fool NLTK
  
## Description
- ./data: directory to store all data (possibly) used
- ./work: directory to save trained models
- predict.py: input a csv file and output static rumor detection results in a json file
- predict_added_to_csv.py: input a csv file and output static rumor detection results to the csv file
- train_20.py: train model using Chinese_Rumor_Dataset
- train_21.py: train model using Chinese_Rumor_Dataset plus crawled data
