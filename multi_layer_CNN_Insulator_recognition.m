clear all;
addpath Common/;
addpath ex1/; % for sigmoid()
addpath CNN1/;
addpath util/;
opts.use_gpu  =false;
%% ex1 Train Convolutional neural network 
%%will run 1 epoch in about 200 second and get around 11% error. 
%%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    %struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 8, 'kernelsize',9) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize',7)%convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 16,'kernelsize',4) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};
%% load training data
load yangben/train_images
load yangben/train_labels
rnd_sel = randperm(size(train_labels,2));
train_images=double(train_images(:,:,:,rnd_sel))/256;
train_labels=train_labels(:,rnd_sel);
if opts.use_gpu
   %opts.gpu_id = auto_select_gpu;
   train_images = gpuArray(double(train_images(:,:,:,rnd_sel))/256);
   train_labels = gpuArray(train_labels(:,rnd_sel));
end
%% init the parameters of training Convolutional neural network 
opts.alpha =0.314;
opts.batchsize =30;
opts.numepochs =30;
load cnn
%cnn = cnnsetup(cnn, train_images, train_labels,opts);
%% Train the Convolutional neural network 
cnn = cnntrain(cnn,train_images,train_labels,opts);
save cnn1 cnn
rnd_sel = randperm(size(train_labels,2));
[acc1, pred1,cnn1] = cnntest(cnn, train_images(:,:,:,rnd_sel(1:500)), train_labels(:,rnd_sel(1:500)),opts);
%% test 
load yangben/test_images
load yangben/test_labels
load cnn
rnd_sel = randperm(size(test_labels,2));
[acc2, pred2,cnn2] = cnntest(cnn, double(test_images(:,:,:,rnd_sel(1:200)))/256, test_labels(:,rnd_sel(1:200)),opts);
%plot mean squared error
%figure; plot(cnn.rL);
%assert(er<0.12, 'Too big error');
