__author__ = 'ANYING'


import numpy as np
from sklearn.metrics import classification_report

import pandas as pd
from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import  SVC



class SVM():
    @staticmethod
    def oversample(X, y):
        """
        用SMOTE算法对负例进行负采样
        """
        model_smote = SMOTE()
        x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X, y)  # 输入数据并作过抽样处理
        smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)  # 按列合并数据框
        groupby_data_smote = smote_resampled.groupby('label').count()  # 对label做分类汇总
        print(groupby_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分类
        return x_smote_resampled, y_smote_resampled

    @staticmethod
    def undersample(X, y):
        #model_RandomUnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
        #x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(X,y)  # 输入数据并作欠抽样处理
        #print(x_RandomUnderSampler_resampled,y_RandomUnderSampler_resampled)
        #return x_RandomUnderSampler_resampled,y_RandomUnderSampler_resampled
        return X,y

    def SVM(self,X, y,X_test,Y_test,type):
        print('begin training')
        if(type=='over'):
            X,y = SVM.oversample(X,y)
        else:
            X,y = SVM.undersample(X,y)
        print('X: ', (X.shape), 'Y : ', np.array(y.shape))
        model = SVC(class_weight='balanced', C=1.0, random_state=0)
        model.fit(X, y)
        print('finish training!')
        print('begin test!')
        y_pred = model.predict(X_test)
        score = model.score(X_test, Y_test)
        print('Score : ', score)
        print(classification_report(Y_test, y_pred))

        return self




