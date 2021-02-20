This repository contains the code for my master_thesis.
It is highly recommended to read this file before running the code to have a better undertsanding it works.

# TOPIC:
  Using deep learning to extract empirical results from tabular data in scientific papers

# PURPOSE OF THE PROJECT:
  The purpose of this project is to collect scientific papers, extract tables, label the dataset and classify tables into four categories.
  Deep neural network, Random Forest classifier, XGBoost, voting classifier were used. 
  Amazon Textract webservice API was called.

# FOLDERS:
  * JSON: Contain the jsons files where the table urls and the table captions will be extracted
  * PNGs: Consists of sub-folders that hold the dataset images 
  
# FILES: 
  * logic.py: This file contains the logic for building a map for the images and its corresponding captions and urls and also responsible for building the tarining dataset.
  * amazon_ws.py: This file contains the functions used from TEXTRACT amazon web service API.
  * feature_extraction.py: A python class that it called by logic.py to extract features out of tables.
  * gridsearchandsoftvoting.py: This file have the model initialization with the grid search pipeline and the voting classifier. 
  * benchmark_model.py: This file contains the classifer for the benchmark model.
  
# PROCESS: 
 (To be able to run the model, it is advised to follow these steps)
  1. Download the json.7z and pngs.7z folders and unzip using 7zip open source program 
  2. Install any of the packages that found in the files (tensorflow, keras, xgboost) uing the commandline *pip install XXX* 
  3. **Start by running the logic.py file**, it will output a tarining_features.txt file on the same directory as the file 
     logic.py will be calling **amazon_ws.py** as well as the **feature_extraction.py** class.
  (*Now the **training dataset is ready** and you can check the file easily by using an excel program as the file is a tab seperator*)
  4. Run the **benchmark_model.py**, The output accuracy of the benchmark model will be showed on the console 
  (The Base model is a DNN using grid search for tuning the hyperparameters) 
  5. Then run the **gridsearchandsoftwoting.py**, The output of the three models results will be presented on the console, also the voting classifier results 
  on both trianing and testing dataset). The time taken for this script to run will be presented in seconds at the end. 
  
