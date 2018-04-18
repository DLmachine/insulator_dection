%% Insulator Recognition
%  Instructions
%  ---------------------------------------------------------------------------
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.
%%============================================================================
%% STEP 0: Initialize Parameters
%  Here we initialize some parameters used for the exercise.
% Configuration
clear all; %prevent out-of-memory errors in Octave on repeated calls
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
addpath Hog_svm/;
addpath multilayer_supervised/;
addpath CNN1/;

% Load Train Data
%data_path='L:\My workshop\machine_learning\data_set\Ñש±¾\';
% As an example, use a single image
%%======================================================================
% Parameters.Note that this controls the number of hierarchical
% Segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo
% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k,and sigma=0.8.
k =500; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.9;
imbuf{1} = imresize(imread('yangben/experiments/2.jpg'),'OutputSize',[600 500]);
%imbuf{2} = imresize(imread('yangben/experiments/12.jpg'),'OutputSize',[600 500]);
%imbuf{3} = imresize(imread('yangben/experiments/14.jpg'),'OutputSize',[600 500]);


% Perform Selective Search
figure
axis tight;  
set(gca, 'box', 'on');
for ch=1:size(imbuf,2)
    
[boxes{ch} blobIndIm blobBoxes hierarchy]=Image2HierarchicalGrouping(imbuf{ch},sigma,k,minSize,colorType,simFunctionHandles);
boxes{ch}  = BoxRemoveDuplicates(boxes{ch});
for  i=1:size(boxes{ch}(:,1))
    qq1=imbuf{ch}(boxes{ch}(i,1):boxes{ch}(i,3),boxes{ch}(i,2):boxes{ch}(i,4),:);
    qq2=imresize(qq1,'OutputSize',[imageDim imageDim]);
    qq{ch}(:,:,i)=single(rgb2gray(qq2))/256;
    c_qq{ch}(:,:,:,i)=double(qq2)/256;
end

%% CNN
load cnn.mat
t_labels=ones(2,size(c_qq{ch},4));
[acc,CNN_Preds{ch}(:),cnn1] = cnntest(cnn,c_qq{ch}, t_labels);
%[unused_, unused_, CNN_Preds]= cnnCost(opttheta,qqq,t_labels,numClasses,filterDim,numFilters,poolDim,true);
A3=find(CNN_Preds{ch}(:)==2); 
%}
%%======================================================================
%% label
subplot(1,size(imbuf,2),ch);
imshow(imbuf{ch});
for  j=1:size(A3,1)
    jno=A3(j);
    rec_l=boxes{ch}(jno,3)-boxes{ch}(jno,1);
    rec_h=boxes{ch}(jno,4)-boxes{ch}(jno,2);
    img=imbuf{ch}(boxes{ch}(jno,1):boxes{ch}(jno,3),boxes{ch}(jno,2):boxes{ch}(jno,4),:);
    %imwrite(img,['ssimg/',num2str(j),'.jpg']); 
    rectangle('Position',[boxes{ch}(jno,2),boxes{ch}(jno,1),rec_h,rec_l],'LineWidth',1,'LineStyle','-','EdgeColor','r');
end;

end