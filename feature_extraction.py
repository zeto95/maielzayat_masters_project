import pandas as pd 
import json 
import pandas as pd
import numpy as np
import textdistance
import re
import os
import ntpath
from pandas.api.types import is_string_dtype
from amazon_ws import get_csv_from_image
from pandas.api.types import is_numeric_dtype
from textdistance import levenshtein

class Feature_Extractor:
    """class instructor """
    def __init__(self):        
        self.types_dict = {}
        self.descriptive_stats_keywords = ['Composite Reliability','AUC','R-square','p-value','t-value','f-value','R2','Estimate','CR','P-value','t-stat','t-statistic','median','Intercept','Specificity','mode','variance','STD Error','confidence','interval','range','Standard Deviation','Percentiles','Descriptive Statistics','Statistics','Coefficient','Count','Coef','average','avg','min','max','maximum','minimum','Variation']
        self.variable_definition_keywords = ['Definitions','definition','description','Explanation','Description','Overview']
        self.model_results_keywords = ['Model','Results','Result','Models']
        
    """
    This function takes CSV and caption as an input and returns calls all other functions to be returned  
    """ 
    def process(self,csv_file,caption):
        try:
            self.df = pd.read_csv(csv_file,header=None, error_bad_lines=False ,sep='\t',encoding="utf8")
            self.df.dropna(axis=1, how='all',inplace= True)  
            no_of_col, no_of_rows = self.df.shape
            empty_cells_ratio = self.get_empty_cells_ratio()
            self.col_dtypes()
            is_coorelation = self.check_coorelation()
            is_caption_descriptive = self.check_caption_descriptive_stats(caption)
            is_cols_descriptive = self.check_cols_descriptive_stats()
            ratio_of_str_int = self.ratio_between_string_and_integers()
            is_cols_var_def = self.check_cols_variable_definition()
            is_cols_model_results = self.check_cols_model_results()
            #return 0,0,0,0,0,0,0,0,0
            return no_of_col,no_of_rows,empty_cells_ratio,is_coorelation,is_caption_descriptive,is_cols_descriptive, ratio_of_str_int,is_cols_var_def,is_cols_model_results
        except Exception as e:
            print('exception in feature', e)

                    
    """ 
    get distance between columns and rows for coorelation matrix 
    check if max distance equal zero then coorelation otherwise returns zero
    """
    def check_coorelation(self):
        try:
            df_row=  pd.DataFrame({'row_names':self.df.iloc[:,0]})
            df_col=  pd.DataFrame({'column_names':self.df.iloc[0,:]})
            frames = [df_row.reset_index(drop=True), df_col.reset_index(drop=True)]
            df_col_rows = pd.concat(frames, axis=1)
            df_col_rows=df_col_rows.dropna()
            df_col_rows['lev']=df_col_rows.apply(lambda x: levenshtein.distance(x['row_names'] , x['column_names']), axis=1)
            if (df_col_rows.lev.max() == 0):
                return 1
            else: 
                return 0 
        except Exception as e:
            return 0

    """ 
    get the ratio between the empty cells and filled ones to check coorealtion
    """
    def get_empty_cells_ratio(self):
        try:
            df = self.df
            df_values = sum(df.apply(lambda x: x.count(), axis=1).values.tolist())
            num_col,num_row = df.shape
            total_count =  num_col * num_row
            empty_cells_count = total_count - df_values
            if total_count == 0:
                return 0
            return empty_cells_count/total_count
        except Exception as e:
            return 0

    """ get columns datatypes """
    def col_dtypes(self):
        #remove first row and pass the new df 
        df_without_header = self.df.iloc[1:]
        for t in df_without_header.dtypes:
            self.types_dict[t] = self.types_dict.get(t,0)+1

    """
    get the ratio between the number of string cells and the intger cells 
    """   
    def ratio_between_string_and_integers(self):
        try:
            str_count = 0
            int_count = 0
            ratio = 0
            df_values = self.df.values.tolist()
            for row in df_values:
                for element in row:
                    if isinstance(element, str):
                        str_count = str_count + 1
                    else:
                        int_count = int_count + 1
            if int_count == 0:
                ratio = 100
            else:
                ratio = str_count/int_count
            return ratio

            # number_of_strings = self.types_dict.get(np.array([object()]).dtype,0) 
            # number_of_int64 = self.types_dict.get( pd.Int64Dtype,0)
            # number_of_float64 = self.types_dict.get(np.array([0.0]).dtype,0)
            # total_number_of_integers = number_of_int64 + number_of_float64
            # if total_number_of_integers == 0 :
            #     ratio =  100            
            # else:   
            #     ratio =  number_of_strings/total_number_of_integers          
            # return ratio
        except Exception as e:
            return 0


    """ 
    check if caption contains any of the descriptive stats keywords
    """
    def check_caption_descriptive_stats(self,caption):
        if re.compile('|'.join(self.descriptive_stats_keywords),re.IGNORECASE).search(caption):
            return 1 
        else:
            return 0
            
    """
    check if column names contain  any of the descriptive stats keywords median/max ...
    """
    def check_cols_descriptive_stats(self):
        cleaned_list = [str(x) for x in self.df.iloc[0,:].tolist() if str(x) != 'nan'] 
        search = ' '.join(cleaned_list)
        if re.compile('|'.join(self.descriptive_stats_keywords),re.IGNORECASE).search(search):
            return 1 
        else:
            return 0
       
    """
    check if column names contain any of the variable definition keywords 
    """
    def check_cols_variable_definition(self):
        cleaned_list = [str(x) for x in self.df.iloc[0,:].tolist() if str(x) != 'nan'] 
        search = ' '.join(cleaned_list)
        if re.compile('|'.join(self.variable_definition_keywords),re.IGNORECASE).search(search):
            return 1 
        else:
            return 0 

    """   
    check if column names contain  any of the model results keywords 
    """
    def check_cols_model_results (self):
        cleaned_list = [str(x) for x in self.df.iloc[0,:].tolist() if str(x) != 'nan'] 
        search = ' '.join(cleaned_list)
        if re.compile('|'.join(self.model_results_keywords),re.IGNORECASE).search(search):
            return 1 
        else:
            return 0 
        




   