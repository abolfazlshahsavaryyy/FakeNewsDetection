from src.data.loader import read_data
from src.features.preprocessing import clean_data
from src.visualization.eda import scatter_featurs
from src.features.split_data import split_data

def main():
    df=read_data("data/Fake.csv",'data/True.csv')
    clean_df=clean_data(df)
    scatter_featurs(clean_data)
    x_train,x_test,x_valid,y_train,y_test,y_valid=split_data(clean_df)


    pass


if __name__ == "__main__":
    main()
