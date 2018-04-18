%function [] = non_dataset_process()
addpath Common/;
addpath ex1/; % for sigmoid()
addpath CNN1/;
addpath util/;
clear all;
Sample_num=200;
imageDim=128;
fcount=0;
cpy_num=1;
%dat_qq=zeros(imageDim,imageDim,3,cpy_num*Sample_num*14);
label_qq=zeros(2,cpy_num*Sample_num*14);
%delete(['bg_train_data/','*.jpg'])
%delete(['bg1/','*.jpg'])
%original1

for yb=1:2
pbuf={'yangben/experiments/bg1/','yangben/experiments/jyz/'}; 
for  i=1:Sample_num
    file_path=[pbuf{yb},num2str(i),'.jpg'];
    res=exist(file_path,'file');
    if(res==2)
        qq1=imread(file_path);
        for  j=1:cpy_num
            fcount=fcount+1;
            %imwrite(qq1,[pbuf{yb},'buf/',num2str(fcount),'.jpg']); 
            dat_qq(:,:,:,fcount)=imresize(qq1,'OutputSize',[imageDim imageDim]);
            label_qq(yb,fcount)=uint8(yb/2);
        end;
    end;
    original_num(yb)=fcount;
end

end
save_path='';
images=double(dat_qq(:,:,:,1:fcount))/256;
labels=label_qq(:,1:fcount);
%% load images labels
%save([save_path,'experiments_labels.mat'],'labels');    
%save([save_path,'experiments_images.mat'],'images');
%{
for  i=1:size(labels,2)
    qq(:,:,i)=double(rgb2gray(images(:,:,:,i)))/256;
end
%}
opts.use_gpu  =false;
load CNN_7_128
[acc1, pred1,cnn1] = cnntest(cnn,images,labels,opts);
save test_CNN1 cnn1

