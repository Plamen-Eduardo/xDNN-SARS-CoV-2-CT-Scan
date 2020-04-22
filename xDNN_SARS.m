clear all
clc


%% Data input

load imdsTrain_SARS
Label =grp2idx(imdsTrain.Labels);
load featuresTrain_SARS

Input1.Features=featuresTrain;
Input1.Labels=grp2idx(imdsTrain.Labels);
Input1.Images=imdsTrain.Files;

tic
Mode='Learning';
[Output1]=xDNN(Input1,Mode);
toc

%% Validation 
load imdsTest_SARS
Label1 =grp2idx(imdsTest.Labels);
load featuresTest_SARS


Input2.xDNNParms=Output1.xDNNParms;
Input2.Images=imdsTest.Files; 
Input2.Features=featuresTest;
Input2.Labels=Label1; 
Mode='Validation'; 
[Output2]=xDNN(Input2,Mode);

disp('Results: ')
Accuracy = Output2.ClasAcc 

precision = Output2.ConfMat(1,1) / (Output2.ConfMat(1,1) + Output2.ConfMat(2,1))
recall = Output2.ConfMat(1,1) / (Output2.ConfMat(1,1) + Output2.ConfMat(1,2))
F1 = (2 * precision * recall) / (precision + recall)

[X,Y,T,AUC] = perfcurve(Input2.Labels,Output2.EstLabs,2); 
AUC

Confusion_Matrix = Output2.ConfMat



