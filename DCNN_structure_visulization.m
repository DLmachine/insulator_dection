addpath CNN1/;
addpath util/;
col=1;
layer_num=3;
col_num=4;
sel=[6,15,19,1];
map_num=[16,12,8];
load test_cnn1
clear A1;
clear v1;

for ly=1:layer_num
    
for  imap=1:map_num(ly)
features=dcnnff(cnn1,ly,imap);%
A1(:,:,1,imap,ly,sel)=features.layers{layer_num+1}.a{1}(:,:,sel);
A1(:,:,2,imap,ly,sel)=features.layers{layer_num+1}.a{2}(:,:,sel);
A1(:,:,3,imap,ly,sel)=features.layers{layer_num+1}.a{3}(:,:,sel);
end

end 

original_img(:,:,1,sel)=features.layers{layer_num+2}.a{1}(:,:,sel);
original_img(:,:,2,sel)=features.layers{layer_num+2}.a{2}(:,:,sel);
original_img(:,:,3,sel)=features.layers{layer_num+2}.a{3}(:,:,sel);

figure;
axis tight;  
set(gca, 'box', 'on');
for  col=1:col_num
   
subplot(layer_num+1,col_num,1+(col-1)*col_num)
imshow(original_img(:,:,:,sel(col)));

for ly=1:layer_num
subplot(layer_num+1,col_num,layer_num+2-ly+(col-1)*col_num) 

clear v1
for i=1:map_num(ly)
v1(:,:,:,i)=A1(:,:,:,i,ly,sel(col));
end
pic=cat(4,v1);
montage(pic);

end 

end
%mean(activations(:,:,1,1),activations(:,:,1,2))
%imshow(activations(:,:,1,1),activations(:,:,2,1))
%imshow(reshape(activations(:,:,1,98),56*4,56*4))