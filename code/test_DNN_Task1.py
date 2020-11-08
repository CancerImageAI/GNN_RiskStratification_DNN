# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:36:55 2019

@author: PC
"""

import numpy as np
from DNN_Task1 import *
import torch
from torch import nn
import os
from torch.autograd import Variable
from skimage import measure
import matplotlib.pyplot as plt
import xlrd
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix
import torch.nn.functional as F
from DataAugmentation import *
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score,matthews_corrcoef
import scipy.stats as stats


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)
    F1_3 = f1_score(truth, predicted)
    print('F1:', F1_3)
    F1_w3 = f1_score(truth, predicted,average='weighted')
    print('F1_weight:',F1_w3)
    MCC3 = matthews_corrcoef(truth, predicted)
    print('MCC:',MCC3)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    
    ResNet_Pretrained_path = './model'
    ResNet_model = ClassifyNet().cuda()
    ResNet_classify_path = os.path.join(ResNet_Pretrained_path, 'DNN.ckpt')#_new_021
    ResNet_modelCheckpoint = torch.load(ResNet_classify_path)
    ResNet_pretrained_dict = ResNet_modelCheckpoint['state_dict']
    ResNet_model_dict = ResNet_model.state_dict()
    ResNet_pretrained_dict = {k: v for k, v in ResNet_pretrained_dict.items() if k in ResNet_model_dict}#filter out unnecessary keys
    ResNet_model_dict.update(ResNet_pretrained_dict)
    ResNet_model.load_state_dict(ResNet_model_dict)
    ResNet_model.eval()
    

    img_path = '../Test/GGO_Crop_64'
    data = xlrd.open_workbook('../Test/GGN_all_64.xls')
    table = data.sheets()[0]
    transform = Transform(32, None)
    Label = []
    ResNet_prob = []
    for i in range(1,len(table.col_values(0))):
        FileName = table.cell_value(i,0)
        test_path = img_path+'/'+FileName+'.npy' 
        data = transform(np.load(test_path))
        data = np.array(data)[np.newaxis,...][np.newaxis,...]
        data = torch.from_numpy(data.astype(np.float32))
        input_data = Variable(data).cuda()
        with torch.no_grad(): 
            ResNet_predict = F.softmax(ResNet_model(input_data), dim = 1)
        ResNet_result = ResNet_predict.data.cpu().numpy()
        
        ResNet_prob.append(ResNet_result[0][1])

        if table.cell_value(i,2) == 'malignant':
            Label.append(1)
        else:
            Label.append(0)
    index_0 = [i for i in range(len(Label)) if Label[i]==0]
    index_1 = [i for i in range(len(Label)) if Label [i]==1]
    index = index_0+index_1
    Label = np.array(Label)[index]
    ResNet_prob = np.array(ResNet_prob)[index]

    print('Test Our ReseNet Model ACC:',accuracy_score(Label,(np.array(ResNet_prob)>0.5).astype(int))*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(ResNet_prob)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(ResNet_prob), np.array(Label))
    print('Test AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')
    prediction_score(Label,(np.array(ResNet_prob)>0.5).astype(int))

           
    LiHaiMing_data = xlrd.open_workbook('../Test/LiHaiMing_Task1.xls')
    LiHaiMing_table = LiHaiMing_data.sheets()[0]
    LiHaiMing_score = LiHaiMing_table.col_values(6)[1:]
    LiHaiMing_score = np.array(LiHaiMing_score)[index]
    LiHaiMing_predict = LiHaiMing_table.col_values(7)[1:]
    LiHaiMing_predict = np.array(LiHaiMing_predict)[index]
    LiHaiMing_predict = [1 if i=='malignant' else 0 for i in LiHaiMing_predict]
    print('LiHaiMing ACC:',accuracy_score(Label,LiHaiMing_predict)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(LiHaiMing_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(LiHaiMing_score), np.array(Label))
    print('LiHaiMing AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']') 
    prediction_score(Label,LiHaiMing_predict)
    
    WangShengPing_data = xlrd.open_workbook('../Test/WangShengPing_Task1.xls')
    WangShengPing_table = WangShengPing_data.sheets()[0]
    WangShengPing_score = WangShengPing_table.col_values(6)[1:]
    WangShengPing_score = np.array(WangShengPing_score)[index]
    WangShengPing_predict = WangShengPing_table.col_values(7)[1:]
    WangShengPing_predict = np.array(WangShengPing_predict)[index]
    WangShengPing_predict = [1 if i=='malignant' else 0 for i in WangShengPing_predict]
    print('WangShengPing ACC:',accuracy_score(Label,WangShengPing_predict)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(WangShengPing_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(WangShengPing_score), np.array(Label))
    print('WangShengPing AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')  
    prediction_score(Label,WangShengPing_predict)
    
    ZhuHui_data = xlrd.open_workbook('../Test/ZhuHui_Task1.xls')
    ZhuHui_table = ZhuHui_data.sheets()[0]
    ZhuHui_score = ZhuHui_table.col_values(6)[1:]
    ZhuHui_score = np.array(ZhuHui_score)[index]
    print('ZhuHui ACC:',accuracy_score(Label,(np.array(ZhuHui_score)>=3).astype(int))*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(ZhuHui_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(ZhuHui_score), np.array(Label))
    print('ZhuHui AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')  
    prediction_score(Label,(np.array(ZhuHui_score)>=3).astype(int))
    
    WangTingTing_data = xlrd.open_workbook('../Test/WangTingTing_Task1.xls')
    WangTingTing_table = WangTingTing_data.sheets()[0]
    WangTingTing_score = WangTingTing_table.col_values(6)[1:]
    WangTingTing_score = np.array(WangTingTing_score)[index]
    WangTingTing_predict = WangTingTing_table.col_values(7)[1:]
    WangTingTing_predict = np.array(WangTingTing_predict)[index]
    WangTingTing_predict = [1 if i=='malignant' else 0 for i in WangTingTing_predict]
    print('WangTingTing ACC:',accuracy_score(Label,WangTingTing_predict)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(WangTingTing_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(WangTingTing_score), np.array(Label))
    print('WangTingTing AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')  
    prediction_score(Label,WangTingTing_predict)
    
    HuTingDan_data = xlrd.open_workbook('../Test/HuTingDan_Task1.xlsx')
    HuTingDan_table = HuTingDan_data.sheets()[0]
    HuTingDan_score = HuTingDan_table.col_values(6)[1:]
    HuTingDan_score = np.array(HuTingDan_score)[index]
    HuTingDan_predict = HuTingDan_table.col_values(7)[1:]
    HuTingDan_predict = np.array(HuTingDan_predict)[index]
    HuTingDan_predict = [1 if i=='malignant' else 0 for i in HuTingDan_predict]
    print('HuTingDan ACC:',accuracy_score(Label,HuTingDan_predict)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(HuTingDan_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(HuTingDan_score), np.array(Label))
    print('HuTingDan AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')  
    prediction_score(Label,HuTingDan_predict)
    
    LiMengLei_data = xlrd.open_workbook('../Test/LiMengLei_Task1.xlsx')
    LiMengLei_table = LiMengLei_data.sheets()[0]
    LiMengLei_score = LiMengLei_table.col_values(6)[1:]
    LiMengLei_score = np.array(LiMengLei_score)[index]
    LiMengLei_predict = LiMengLei_table.col_values(7)[1:]
    LiMengLei_predict = np.array(LiMengLei_predict)[index]
    LiMengLei_predict = [1 if i=='malignant' else 0 for i in LiMengLei_predict]
    print('LiMengLei ACC:',accuracy_score(Label,LiMengLei_predict)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(Label),np.array(LiMengLei_score)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(LiMengLei_score), np.array(Label))
    print('LiMengLei AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')  
    prediction_score(Label,LiMengLei_predict)
    
    ZhuHui_predict = (np.array(ZhuHui_score)>=3).astype(int)
    DNN_predict = (np.array(ResNet_prob)>0.5).astype(int)
    print('Kappa Value')
    print('DNN and GT:', cohen_kappa_score(DNN_predict,Label))
    print('WSP and DNN:', cohen_kappa_score(DNN_predict,WangShengPing_predict))
    print('WSP and Hist:', cohen_kappa_score(Label,WangShengPing_predict))
    print('WSP and ZHH:', cohen_kappa_score(WangShengPing_predict,ZhuHui_predict))
    print('WSP and LHM:', cohen_kappa_score(WangShengPing_predict,LiHaiMing_predict))
    print('WSP and WTT:', cohen_kappa_score(WangShengPing_predict,WangTingTing_predict))
    print('WSP and HTD:', cohen_kappa_score(WangShengPing_predict,HuTingDan_predict))
    print('WSP and LML:', cohen_kappa_score(WangShengPing_predict,LiMengLei_predict))
    print('ZHH and DNN:', cohen_kappa_score(DNN_predict,ZhuHui_predict))
    print('ZHH and Hist:', cohen_kappa_score(Label,ZhuHui_predict))
    print('ZHH and LHM:', cohen_kappa_score(ZhuHui_predict, LiHaiMing_predict))
    print('ZHH and WTT:', cohen_kappa_score(ZhuHui_predict, WangTingTing_predict))
    print('ZHH and HTD:', cohen_kappa_score(ZhuHui_predict, HuTingDan_predict))
    print('ZHH and LML:', cohen_kappa_score(ZhuHui_predict, LiMengLei_predict))
    print('LHM and DNN:', cohen_kappa_score(DNN_predict,LiHaiMing_predict))
    print('LHM and Hist:', cohen_kappa_score(Label,LiHaiMing_predict))
    print('LHM and WTT:', cohen_kappa_score(LiHaiMing_predict, WangTingTing_predict))
    print('LHM and HTD:', cohen_kappa_score(LiHaiMing_predict, HuTingDan_predict))
    print('LHM and LML:', cohen_kappa_score(LiHaiMing_predict, LiMengLei_predict))
    print('WTT and DNN:', cohen_kappa_score(DNN_predict,WangTingTing_predict))
    print('WTT and Hist:', cohen_kappa_score(Label,WangTingTing_predict))
    print('WTT and HTD:', cohen_kappa_score(WangTingTing_predict, HuTingDan_predict))
    print('WTT and LML:', cohen_kappa_score(WangTingTing_predict, LiMengLei_predict))
    print('HTD and DNN:', cohen_kappa_score(DNN_predict,HuTingDan_predict))
    print('HTD and Hist:', cohen_kappa_score(Label,HuTingDan_predict))
    print('HTD and LML:', cohen_kappa_score(HuTingDan_predict, LiMengLei_predict))
    print('LML and DNN:', cohen_kappa_score(DNN_predict,LiMengLei_predict))
    print('LML and Hist:', cohen_kappa_score(Label,LiMengLei_predict))
    

    
    