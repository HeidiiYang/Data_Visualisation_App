import joblib
 
def predict(data):
    logreg=joblib.load('logreg_model.sav')
    return logreg.predict(data)
