clear all;
load test_CNN1
col=1;
col_num=4;
sel=[1,3,6,7];
map_num=[8,12,16];
layer_num=4;

figure;

for  col=1:col_num

subplot(layer_num,col_num,col)
original_img(:,:,1)=cnn1.layers{1,1}.a{1,1}(:,:,sel(col));
original_img(:,:,2)=cnn1.layers{1,1}.a{1,2}(:,:,sel(col));
original_img(:,:,3)=cnn1.layers{1,1}.a{1,3}(:,:,sel(col));
imshow(original_img);

subplot(layer_num,col_num,col+col_num)
layer=3;
clear v1
for i=1:map_num(1)
v1(:,:,1,i)=cnn1.layers{layer,1}.a{1,i}(:,:,sel(col));
end
pic=cat(2,v1);
montage(pic)

subplot(layer_num,col_num,col+2*col_num)
layer=5;
clear v1
for i=1:map_num(2)
v1(:,:,1,i)=cnn1.layers{layer,1}.a{1,i}(:,:,sel(col));
end
pic=cat(2,v1);
montage(pic)

subplot(layer_num,col_num,col+3*col_num)
layer=7;
clear v1
for i=1:map_num(3)
v1(:,:,1,i)=cnn1.layers{layer,1}.a{1,i}(:,:,sel(col));
end
pic=cat(2,v1);
montage(pic)


end