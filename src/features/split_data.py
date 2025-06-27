from sklearn.model_selection import train_test_split
from src.const.constant_valud import *

def split_train_test_valid(x_df,y_df,test_size,valid_size):
    x_train,x_test,y_train,y_test=train_test_split(x_df,y_df,test_size=test_size,random_state=RANDOM_STATE)
    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=valid_size,random_state=RANDOM_STATE)
    return x_train,x_test,x_valid,y_train,y_test,y_valid

def split_data(clean_df):
    '''
        input : clean df : pd.Dataframe

        output : x_train,x_test,x_valid,y_train,y_test,y_valid

        
    '''
    x_df=clean_df.drop(columns=['target'])
    y_df=clean_df["target"]
    return split_train_test_valid(x_df,y_df,test_size=0.2,valid_size=0.25) #x_train,x_test,y_train,y_test