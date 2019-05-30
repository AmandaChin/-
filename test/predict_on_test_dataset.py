from sklearn.metrics import classification_report

from dataset.read_dataset import get_dataset_dataframe
from training.train import extract_training_data_from_dataframe, trained_model_pickle_file
import pandas as pd
import os
def predict(model):
    df = get_dataset_dataframe(directory=os.path.expanduser('D:/ay/DDIExtraction-svm/dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/'))
    X, Y = extract_training_data_from_dataframe(df)
    # model = pd.read_pickle(trained_model_pickle_file)
    y_pred  = model.predict(X)
    score = model.score(X, Y)
    print('Score : ', score)
    print(classification_report(Y, y_pred))



