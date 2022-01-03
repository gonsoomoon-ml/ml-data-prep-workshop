######################################
# 데이터 전처리 관련 함수
######################################
def change_y(df, col, isChange=False):
    '''
    isChange 이면 레이블 컬럼을 데이터 프레임의 가장 맨 앞에 위치 함.
    그렇지 않으면 'True.' --> 1 로 바꾸고, 아니면 0으로 합니다.
    '''
    ldf = df.copy()
    if isChange:
        y = np.where(ldf[col] == 'True.', 1, 0)
    else:
        y = df[col]
        
    ldf.drop(columns=col, inplace=True) # 기존 레이블 컬럼 삭제
    ldf.insert(0, value=y, column=col)

    return ldf




######################################
# 모델 평가 관련 함수
######################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")



def get_binary_prediction(df, threshold):
    return np.where(df >= threshold, 1, 0)


from sklearn.metrics import confusion_matrix

def plot_cm2(target_train, train_pred, target_valid, valid_pred, title, figsize=(8,3)):
    # Building the 2 confusion matrices train/valid
    
    def cm_calc(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        return cm, annot

    
    # Building the confusion matrices
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)

    # Training data
    ax = axes[0]
    ax.set_title("for training data")
    cm0, annot0 = cm_calc(target_train, train_pred)    
    sns.heatmap(cm0, cmap= "YlGnBu", annot=annot0, fmt='', ax=ax)
    
    # Validation data
    ax = axes[1]
    ax.set_title("for validation data")
    cm1, annot1 = cm_calc(target_valid, valid_pred)
    sns.heatmap(cm1, cmap= "YlGnBu", annot=annot1, fmt='', ax=ax)

    
    fig.suptitle(title, y=1.05)
    plt.show()
    
from sklearn.metrics import roc_curve, classification_report

def roc_auc_plot(model, train, target_train, valid, target_valid):
    # Draw ROC-UAC plots and print report: precision, recall, f1-score, support

    def roc_plot(target, pred_prob, title):
        # Calc and draw ROC plot for df and target
        sns.set(font_scale=1.5)
        sns.set_color_codes("muted")
        plt.figure(figsize=(5, 4))
        fpr, tpr, thresholds = roc_curve(target, pred_prob, pos_label=1)
        plt.plot(fpr, tpr, lw=2, label='ROC curve ')
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.show()        

    # Report
    train_pred_prob = model.predict(train)    
    train_pred = get_binary_prediction(train_pred_prob, threshold=0.5)
    valid_pred_prob = model.predict(valid)    
    valid_pred = get_binary_prediction(valid_pred_prob, threshold=0.5)


    print('Classification report for training data\n',
          classification_report(target_train, train_pred, target_names=['0', '1']))
    print('Classification report for validation data\n',
          classification_report(target_valid, valid_pred, target_names=['0', '1']))

    roc_plot(target_train, train_pred_prob, "ROC curve for training data")
    roc_plot(target_valid, valid_pred_prob, "ROC curve for validation data")    
    
import shap    
    
def data_force_plot_n(model, X):

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

######################################
# AutoGluon 함수
######################################


import numpy as np
def get_prediction_set(prediction,prediction_prob , threshold):
    df = prediction_prob.copy()
    df['score'] = df[1] * 1000
    df['score'] = df['score'].astype('int')
    df.drop(columns=[0,1], inplace=True)
    
    df['pred'] = np.where(df['score'] >= threshold, 1,0)
    cols = ['score','pred']
    
    df = df[cols]

    return df

from sklearn.metrics import classification_report, roc_auc_score

def compute_f1(y_true, y_pred):
    print("- ROC_AUC SCORE")
    print(f'\t-{round(roc_auc_score(y_true, y_pred),3)}')
    print("\n- F1 SCORE")    
    print(classification_report(y_true = y_true, y_pred = y_pred))    
    
    cm = confusion_matrix(y_true= y_true, y_pred= y_pred)

    print(cm)





    