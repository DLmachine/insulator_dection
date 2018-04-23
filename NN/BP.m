clear all;
close all;
%<----------------------------!运用BP网络进行图像分类--------------------------->
%读入样本1,                     图的红色区域，代表城市，期望输出：[1;0;0]   
data_path='L:\My workshop\machine_learning\data_set\样本\oo\';
I=imread([data_path,'23.jpg']);
%将样本图像降维处理
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%灰度值归一化
R=im2double(R);
G=im2double(G);
B=im2double(B);
[M N]=size(R);
R=reshape(R',[1 M*N]);
G=reshape(G',[1 M*N]);
B=reshape(B',[1 M*N]);
%初始化输入矢量P和输出矢量T
P=[];
T=[];
P=[R;G;B];
T=[1;0;0];
[m n]=size(P);
T=concur(T,n);

%读入样本图像2                  图的绿色区域，代表城市，期望输出：[0,1,0]   
I=imread([data_path,'32.jpg']);
%将样本图像降维处理
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%灰度值归一化
R=im2double(R);
G=im2double(G);
B=im2double(B);
[M N]=size(R);
R=reshape(R',[1 M*N]);
G=reshape(G',[1 M*N]);
B=reshape(B',[1 M*N]);
P1=[R;G;B];
T1=[0;1;0];
P=[P,P1];
[m n]=size(P1);
T1=concur(T1,n);
T=[T,T1];

%读入样本图像3                图的蓝色区域，代表城市，期望输出：[0;0;1]
I=imread([data_path,'61.jpg']);
%将样本图像降维处理
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%灰度值归一化
R=im2double(R);
G=im2double(G);
B=im2double(B);
[M N]=size(R);
R=reshape(R',[1 M*N]);
G=reshape(G',[1 M*N]);
B=reshape(B',[1 M*N]);
P1=[R;G;B];
T1=[0;0;1];
P=[P,P1];
[m n]=size(P1);
T1=concur(T1,n);
T=[T,T1];

%创建一个前向神经网络
net=newff(minmax(P),[5,3],{'logsig','purelin'},'traingdx');
%设置训练参数
net.trainParam.show=50;
net.trainParam.epochs=100;   %最大训练步数为1000
net.trainParam.goal=0.001;
net=init(net);
%对BP网络进行训练
net=train(net,P,T);


%读入待分类遥感图像
  imread([data_path,'12.jpg']);
  figure,imshow(I);
  %将彩色图像降维
  R=I(:,:,1);
  G=I(:,:,2);
  B=I(:,:,3);
  %将灰度值归一化处理
  R=im2double(R);
  G=im2double(G);
  B=im2double(B);
  [M,N]=size(R);
  R=reshape(R',[1 M*N]);
  G=reshape(G',[1 M*N]);
  B=reshape(B',[1 M*N]);
  p=[R;G;B];
  %对BP网络进行仿真
  Y=sim(net,p);
  R=Y(1,:);
  X=R;              
  classR=[];    
  for i=0:(M-1)
      classR=[classR;R((i*N+1):(i*N+N))];
  end
  G=Y(2,:);
  classG=[];
  for i=0:(M-1)
      classG=[classG;G((i*N+1):(i*N+N))];
  end
  B=Y(3,:);
  classB=[];
  for i=0:(M-1)
      classB=[classB;B((i*N+1):(i*N+N))];
  end

  R=abs(classR)*255;
  R=uint8(R);
  G=abs(classG)*255;
  G=uint8(G);
  B=abs(classB)*255;
  B=uint8(B);
  classify=cat(3,R,G,B);
  figure,imshow(classify);

