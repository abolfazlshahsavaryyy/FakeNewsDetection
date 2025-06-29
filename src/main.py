from src.read_data.loader import read_data
from src.features.preprocessing import clean_data
from src.visualization.eda import scatter_featurs
from src.features.split_data import split_data
from src.models.hyperparameter_tuning.hyperparameter_tuning_svc import manual_grid_search_linear
from src.config.config import settings
from src.models.train_model.LogisticRegression_model import LogisticRegression_model 
from src.models.train_model.SVC_model import SVC_model
from src.models.predict.predict_logistic_regression import predict_logistic_regression
from src.models.predict.predict_svc import predict_svc
from src.model_save.save_models import save_model
from src.read_data.loader import read_clean_data

def main():

    print("src main.py has been run :)")
    # print("Read data")
    # df=read_data("data/Fake.csv",'data/True.csv')
    # print("clean data")
    # clean_df=clean_data(df)
    print("read the clean data")
    clean_df=read_clean_data('/home/abolfazl/Documents/python-code/FakeNewsDetection/data/clean_data.csv')
    #print("scatter plot the featurs")
    #scatter_featurs(clean_df)
    x_train,x_test,x_valid,y_train,y_test,y_valid=split_data(clean_df)
    
    #best_model, best_params, best_score=manual_grid_search_linear(x_train,y_train,x_valid,y_valid,settings.PARAM_GRID_LOGISTIC)
    print("train svc model")
    model_svc=SVC_model(x_train,y_train,settings.svc_parameter)
    print("trian logistic regression model")
    model_logistic_regression=LogisticRegression_model(x_train,y_train,settings.logistic_regression_paramater)

    print(f"Test accuracy logistic regression:{predict_logistic_regression
                                               (
                                                   model_logistic_regression,x_test,y_test
                                                   )}")
    print(f"Test accuracy SVC :{predict_svc(
        model_svc,x_test,y_test
    )}")
    save_model([model_svc,model_logistic_regression])



    pass


if __name__ == "__main__":
    main()
