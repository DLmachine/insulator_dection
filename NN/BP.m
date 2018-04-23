clear all;
close all;
%<----------------------------!����BP�������ͼ�����--------------------------->
%��������1,                     ͼ�ĺ�ɫ���򣬴�����У����������[1;0;0]   
data_path='L:\My workshop\machine_learning\data_set\����\oo\';
I=imread([data_path,'23.jpg']);
%������ͼ��ά����
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%�Ҷ�ֵ��һ��
R=im2double(R);
G=im2double(G);
B=im2double(B);
[M N]=size(R);
R=reshape(R',[1 M*N]);
G=reshape(G',[1 M*N]);
B=reshape(B',[1 M*N]);
%��ʼ������ʸ��P�����ʸ��T
P=[];
T=[];
P=[R;G;B];
T=[1;0;0];
[m n]=size(P);
T=concur(T,n);

%��������ͼ��2                  ͼ����ɫ���򣬴�����У����������[0,1,0]   
I=imread([data_path,'32.jpg']);
%������ͼ��ά����
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%�Ҷ�ֵ��һ��
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

%��������ͼ��3                ͼ����ɫ���򣬴�����У����������[0;0;1]
I=imread([data_path,'61.jpg']);
%������ͼ��ά����
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%�Ҷ�ֵ��һ��
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

%����һ��ǰ��������
net=newff(minmax(P),[5,3],{'logsig','purelin'},'traingdx');
%����ѵ������
net.trainParam.show=50;
net.trainParam.epochs=100;   %���ѵ������Ϊ1000
net.trainParam.goal=0.001;
net=init(net);
%��BP�������ѵ��
net=train(net,P,T);


%���������ң��ͼ��
  imread([data_path,'12.jpg']);
  figure,imshow(I);
  %����ɫͼ��ά
  R=I(:,:,1);
  G=I(:,:,2);
  B=I(:,:,3);
  %���Ҷ�ֵ��һ������
  R=im2double(R);
  G=im2double(G);
  B=im2double(B);
  [M,N]=size(R);
  R=reshape(R',[1 M*N]);
  G=reshape(G',[1 M*N]);
  B=reshape(B',[1 M*N]);
  p=[R;G;B];
  %��BP������з���
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

