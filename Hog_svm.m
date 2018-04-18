clc; 
clear ;  
addpath Hog_svm/;
use_gpu  = false;
%% 训练阶段  
load yangben/original_train_images
load yangben/original_train_labels
if use_gpu
   train_images = gpuArray(train_images);
   train_labels = gpuArray(train_labels);
end

for  i=1:size(train_labels,2)
    qq(:,:,i)=double(rgb2gray(train_images(:,:,:,i)))/256;
end
sample_num=size(qq,3);
rnd_sel = randperm(sample_num);

%读取负样本并计算hog特征  
for i=1:sample_num/2
   hog =hogcalculator(qq(:,:,i));  
   data(i,:)=hog;  
   disp(['epoch ', num2str(i)]);
end  

%读取正样本并计算hog特征  
for i=1:sample_num/2
   hog =hogcalculator(qq(:,:,i+sample_num/2));  
   data(sample_num/2+i,:)=hog;  
   disp(['epoch ', num2str(i)]);
end  

[train, test] = crossvalind('holdOut',train_labels(2,:));  
cp = classperf(train_labels(2,:));
svmStruct = svmtrain(data(train,:),train_labels(2,train)); 
save Common/svmStruct svmStruct  
train_classes = svmclassify(svmStruct,data(test,:));
classperf(cp,train_classes,test);  
fprintf('train accuracy is %g\n', cp.CorrectRate);

%% 训练完成后保存 svmStruct即可对新输入的对象进行分类了无需再执行上面训练阶段代码  
load Common/svmStruct  
load yangben/test_images
load yangben/test_labels
for  j=1:200
    hogt =hogcalculator(double(rgb2gray(test_images(:,:,:,j)))/256);  
    classes_test(j,1) = svmclassify(svmStruct,hogt);%classes的值即为分类结果  
end
fprintf('test accuracy is %g\n', mean(classes_test == test_labels(2,1:200)'));