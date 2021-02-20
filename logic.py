import json
import os
import path
import ntpath
import pandas as pd
from amazon_ws import get_csv_from_image
from feature_extraction import Feature_Extractor  

#Global variables
PNGS_FOLDER_PATH = 'pngs'
JSON_FILES_PATH = 'Json'

"""
This function builds a map that contains all images with their path and labels.
"""
def build_image_map (pngs_folder):  
    pngs_dict = {}
    for folder in os.listdir(pngs_folder):
        for filename in os.listdir(os.path.join(pngs_folder,folder)):
            pngs_dict[filename]=(folder,os.path.join(pngs_folder,folder,filename))
    return pngs_dict

"""
This function takes the JSON file as input and returns the tables only and gives back the caption and url.
"""     
def get_caption (json_file, pngs_dict):
    caption_dict = list()
    try:
        with open(json_file, encoding="ISO-8859-1") as f:
            data = json.load(f)   
    except Exception as e:
        print('Exception: ', e)                
    for json_object in data:
        if json_object['figType'] == "Table":
            render_url = json_object['renderURL']
            image_name = ntpath.basename(render_url)
            try:
                url = pngs_dict[image_name][1]
            except: 
                continue
            label = pngs_dict[image_name][0]
            caption = json_object['caption']
            caption_dict.append((caption,url,label))   
                 
    return caption_dict

"""
Extract the features and write them into csv file. Extract features as a list that contain extracted features and the corresponding labels.
"""    
def extract_features(all_captions_url):
    ft_ext = Feature_Extractor()
    features = list()
## train model with all json and pngs 
    with open('Training_features.txt', 'w') as f:
        # Headers to be insterted in the training features CSV file 
        headers = ['Image_Url', 'Number_of_columns','Number_of_rows','empty_cells_ratio','Is_coorelation','Is_caption_descriptive','Is_columns_descriptive','ratio_str_vs_int','is_cols_var_def','is_cols_model_results','Label']
        f.write('\t'.join(headers) + '\n')
        for file in all_captions_url:  
            try:  
                for entry in file:
                    caption = entry[0]
                    url = entry[1]
                    label = entry[2]                  
                    csv_file = get_csv_from_image(url)   
                    no_of_col,no_of_rows,empty_cells_ratio,is_coorelation,is_caption_descriptive,is_cols_descriptive,ratio_of_str_int,is_cols_var_def,is_cols_model_results  = ft_ext.process(csv_file,caption)
                    # features.append((*extracted_features,label))    
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( url,no_of_col,no_of_rows,empty_cells_ratio,is_coorelation,is_caption_descriptive,is_cols_descriptive,ratio_of_str_int,is_cols_var_def,is_cols_model_results,label))
                    f.flush()
                    os.fsync(f.fileno())          
            except Exception as e:
                print(file, " Parsing file failed", e)
    f.close()

def main():
    pngs_map = build_image_map(PNGS_FOLDER_PATH)
    all_captions_url = list()
    for filename in os.listdir(JSON_FILES_PATH):
        all_captions_url.append(get_caption(os.path.join(JSON_FILES_PATH,filename), pngs_map))
    extract_features(all_captions_url)

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print('Error: ' + str(ex))
        exit(777)