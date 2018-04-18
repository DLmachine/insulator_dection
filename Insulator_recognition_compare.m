%% Insulator Recognition
%  Instructions
%  -----------------------------------
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.
%%======================================================================
%% STEP 0: Initialize Parameters
%  Here we initialize some parameters used for the exercise.
% Configuration
clear all; %prevent out-of-memory errors in Octave on repeated calls
close all;
imageDim = 128;
numClasses = 2;   % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 16;  % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)
addpath common/;
addpath util/;
addpath ex1/; % for sigmoid()
addpath('SelectiveSearchCodeIJCV/Dependencies');
addpath SelectiveSearchCodeIJCV/;
addpath edgebox/release;
addpath Hog_svm/;
addpath multilayer_supervised/;

% Load Train Data
%data_path='L:\My workshop\machine_learning\data_set\样本\';
% As an example, use a single image
%%======================================================================
%% 选择性搜索
% Parameters.Note that this controls the number of hierarchical
% Segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo
% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k,and sigma=0.8.
k =400; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.9;
im = imread('yangben/experiments/10.jpg');
% Perform Selective Search
%[boxes blobIndIm blobBoxes hierarchy]=Image2HierarchicalGrouping(im,sigma,k,minSize,colorType,simFunctionHandles);
%boxes = BoxRemoveDuplicates(boxes);

% Perform edgebox
boxes=boxesproposal(im);
boxes=boxes(boxes(:,5)>0.12,:);
boxes(:,[3,4])=boxes(:,[4,3]);
boxes(:,[1,2])=boxes(:,[2,1]);
boxes(:,3)=boxes(:,1)+boxes(:,3);
boxes(:,4)=boxes(:,2)+boxes(:,4);

for  i=1:size(boxes(:,1))
    qq1=im(boxes(i,1):boxes(i,3),boxes(i,2):boxes(i,4),:);
    qq2=imresize(qq1,'OutputSize',[imageDim imageDim]);
    qq(:,:,i)=single(rgb2gray(qq2))/256;
    c_qq(:,:,:,i)=double(qq2)/256;
end
%%======================================================================
%% SVM+HOG
load Common/svmStruct  
for  j=1:size(qq,3)
    hogt =hogcalculator(qq(:,:,j));  
    classes_res(j,1) = svmclassify(svmStruct,hogt);%#ok<SVMCLASSIFY> %classes的值即为分类结果  
end
A1=find(classes_res(:)==1); %记录位置
%%======================================================================
%% PCA+NN
load Common/PCA_COEFF
load common/opt_ei
load common/opt_params
input_data=COEFF'*reshape(qq,[128*128,size(qq,3)]);
for  i=1:size(input_data,2)
    dis=max(input_data(:,i))-min(input_data(:,i));
    input_data(:,i)=(input_data(:,i)-min(input_data(:,i))*ones(size(input_data(:,i))))/dis;
end
[unused_, unused_, pred] = supervised_dnn_cost( opt_params, ei,input_data, [], true);
[unused_,classes_res] = max(pred);
A2=find(classes_res(:)==1); %记录位置
%%======================================================================
%% CNN
addpath CNN1/;
load cnn.mat
opts.use_gpu  =false;
t_labels=ones(2,size(c_qq,4));
[~,CNN_Preds,cnn1] = cnntest(cnn,c_qq, t_labels,opts);
A3=find(cnn1.o(2,:)>=0.7)'; %记录位置
%%======================================================================
%% 绝缘子标记
imshow(im)
for  j=1:size(A3,1)
    jno=A3(j);
    rec_l=boxes(jno,3)-boxes(jno,1);
    rec_h=boxes(jno,4)-boxes(jno,2);
    img=im(boxes(jno,1):boxes(jno,3),boxes(jno,2):boxes(jno,4),:);
    %imwrite(img,['ssimg/',num2str(j),'.jpg']); 
    rectangle('Position',[boxes(jno,2),boxes(jno,1),rec_h,rec_l],'LineWidth',2,'LineStyle','-','EdgeColor','r')
end
