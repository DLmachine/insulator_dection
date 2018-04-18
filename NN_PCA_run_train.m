% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function
%% setup environment
clear all; % prevents crashing on reloading 45 MB training data from disk
% experiment information
% a struct containing network layer sizes etc
ei = [];
% add common directory to your path for
% minfunc and mnist data helpers
addpath Common;
addpath multilayer_supervised;
addpath(genpath('Common/minFunc_2012/minFunc'));

%% load training data
load yangben/train_images
load yangben/train_labels
for  i=1:size(train_labels,2)
    qq(:,:,i)=double(rgb2gray(train_images(:,:,:,i)))/256;
end
train_num=size(train_labels,2);
rnd_sel = randperm(train_num);
data_train=qq(:,:,rnd_sel);
% 拉成列向量
data_train=reshape(data_train,[128*128,train_num]);
labels_train=train_labels(2,rnd_sel);
labels_train=labels_train';
labels_train(labels_train==0)=2;

%% PCA 降维
[COEFF,SCORE, latent]=pca(data_train');
save common/PCA_COEFF COEFF
load common/PCA_COEFF
data_train=SCORE';
%A1=cumsum(latent)./sum(latent);
%data_train=data_train(A1>0.85,:);
%pareto(latent);
% 正规化
for  i=1:train_num
    dis=max(data_train(:,i))-min(data_train(:,i));
    data_train(:,i)=(data_train(:,i)-min(data_train(:,i))*ones(size(data_train(:,i))))/dis;
end

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce 100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)
% dimension of input features
ei.input_dim = size(data_train,1);
% number of output classes [fixed for digits task]
ei.output_dim = 2;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256,64,ei.output_dim]; % default
% scaling parameter for l2 weight regularization penalty
ei.lambda = 3;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function     % this was EASY relative to all the NN bookkeeping
ei.activation_fun = 'logistic'; % 'logistic', 'tanh', or 'rectified'
% toggle my paranoid error checking
ei.DEBUG = false; 
if ei.DEBUG
    % speed things up for debugging
    m = size(data_train, 2)/1000;
    data_train = data_train(:,1:m);
    labels_train = labels_train(1:m);
end
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.useMex = true; % my additions
options.MaxIter = 600; % add this? it's running like hours without this option... my kingdom for a pickle
if ei.DEBUG; options.DerivativeCheck = 'on'; end

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
save common/opt_ei ei
save common/opt_params opt_params
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[unused_, unused_, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[unused_,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);

%% load testing data
load yangben/test_images
load yangben/test_labels
test_num=size(test_labels,2);
rnd_sel = randperm(test_num);
for  i=1:test_num
    qqqq(:,:,i)=double(rgb2gray(test_images(:,:,:,i)))/256;
end
t_images=qqqq(:,:,rnd_sel);
t_images=COEFF'*reshape(t_images,[128*128,test_num]);
for  i=1:test_num
    dis=max(t_images(:,i))-min(t_images(:,i));
    t_images(:,i)=(t_images(:,i)-min(t_images(:,i))*ones(size(t_images(:,i))))/dis;
end
t_labels=test_labels(2,rnd_sel)';
t_labels(t_labels==0)=2;

[unused_, unused_, pred] = supervised_dnn_cost( opt_params, ei, t_images, [], true);
[unused_,pred] = max(pred);
acc_test = mean(pred'==t_labels);
fprintf('test accuracy: %f\n', acc_test);
