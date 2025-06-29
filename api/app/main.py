from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import joblib
from FakeNewsDetection.api.app.request_model import News
app = FastAPI()

# Load your trained model
print("load the logistic regression model")
lr_model = joblib.load('/home/abolfazl/Documents/python-code/FakeNewsDetection/data/model/LogisticRegression(max_iter=1000).pkl')
print("load svc model ")
svc_model= joblib.load('/home/abolfazl/Documents/python-code/FakeNewsDetection/data/model/LinearSVC(C=0.01).pkl')
class NewsRequest(BaseModel):
    title: str
    text: str

@app.post("/predict/logistic-regression")
def predict(news: NewsRequest):
    print("get the input")
    news_instance = News(news.title, news.text)
    print("predict the label")
    prediction = news_instance.predict_logistic_regression(lr_model)
    return {"prediction": prediction}

@app.post("/predict/svc")
def predict(news: NewsRequest):
    news_instance = News(news.title, news.text)
    prediction = news_instance.predict_svc(svc_model)
    return {"prediction": prediction}
