addpath Deconvolution/;
addpath SuperPixel/;
addpath CNN1/;
close all;
clear all;
load test_CNN1
sel=19;
oringinal_img(:,:,1)=cnn1.layers{1,1}.a{1,1}(:,:,sel);
oringinal_img(:,:,2)=cnn1.layers{1,1}.a{1,2}(:,:,sel);
oringinal_img(:,:,3)=cnn1.layers{1,1}.a{1,3}(:,:,sel);

%{
for map=1:numel(cnn1.layers{layer,1}.a)
%A(:,:,map)=cnn1.layers{layer,1}.a{1,map}(:,:,sel);
%A(:,:,map)=deconvSps(cnn1.layers{layer,1}.a{1,map}(:,:,sel),cnn1.layers{layer,1}.k{1,1}{1,map},0.2,10);
end
%}

%{
features=dcnnff(cnn1,3,6);
A(:,:,1)=features.layers{4}.a{1}(:,:,sel);
A(:,:,2)=features.layers{4}.a{2}(:,:,sel);
A(:,:,3)=features.layers{4}.a{3}(:,:,sel);
%}

layer=3;
A(:,:,1)=cnn1.layers{layer,1}.a{1,5}(:,:,sel);
A(:,:,2)=cnn1.layers{layer,1}.a{1,5}(:,:,sel);
A(:,:,3)=cnn1.layers{layer,1}.a{1,5}(:,:,sel);
A(A<0.94)=0;

% ???
rga=imresize(oringinal_img(:,:,1),'OutputSize',[size(A,1),size(A,2)]);
pbuf={'yangben/experiments/bg/','yangben/experiments/jyz/'}; 
file_path=[pbuf{2},num2str(sel),'.jpg'];
qq1=double(imread(file_path))/256;
res1=imresize(A,'OutputSize',size(qq1(:,:,1)));
%res1=imresize(cnn1.layers{layer,1}.a{1,3}(:,:,sel),'OutputSize',size(qq1(:,:,1)));
res1=res1.*qq1;
imwrite(res1,[pbuf{2},'res1.jpg']);

res2=SLIC(res1,10);
imwrite(res2,[pbuf{2},'res2.jpg']);

res3=res1.*res2;
imwrite(rgb2gray(res3),[pbuf{2},'res3.jpg']);
imshow([res1,res2,res3]);

