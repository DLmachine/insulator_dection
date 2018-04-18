function net = cnnff(net, x,opts)%���ѵ����ǰ����̣�  
    n = numel(net.layers);
    net.layers{1}.a{1} =reshape(x(:,:,1,:),[size(x,1),size(x,2),size(x,4)]); % a������map,��һ����128,128,50���ľ���
    net.layers{1}.a{2} =reshape(x(:,:,2,:),[size(x,1),size(x,2),size(x,4)]);
    net.layers{1}.a{3} =reshape(x(:,:,3,:),[size(x,1),size(x,2),size(x,4)]); 
    inputmaps = 3;
    for l = 2 : n   %���ÿһ�������
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps   % ��Ըò��ÿһ��feture map  
                %create temp output map
                %z = zeros(size(net.layers{l-1}.a{1}));
                z = zeros(size(net.layers{l - 1}.a{1})-[net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                if(opts.use_gpu==true) 
                z=gpuArray(z);end;
                % �ò�feture map�Ĵ�С�����һλ������ͼƬ���� ��ʼ��Ϊ0  
                for i = 1 : inputmaps  %���ÿһ������feature map
                    z = z + convn(net.layers{l-1}.a{i}, net.layers{l}.k{i}{j}, 'valid');    
                    %���������  k{i}{j} ��5*5��double���ͣ�����a{i}������ͼƬ��feature map ��СΪ28*28*50 50Ϊͼ������  
                    %������������k�Ѿ���ת��180�ȣ�z����ľ��Ǹò������е�����
                end
                % ͨ�������Լ�ƫֵ
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});%%��ȡsigmoid function��ֵ 
            end
            inputmaps = net.layers{l}.outputmaps; %% �����µ�����feature maps�ĸ���      
        elseif strcmp(net.layers{l}.type, 's') %�²������õķ�����,2*2��ӳ���Ȩֵ1/4,  û��ȡƫ�ú�ȡsigmoid  
            %  downsample
            for j = 1 : inputmaps
                %����������mean-pooling
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');  
                 %�Ⱦ������и���ȡ���
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                 %�õ��Ľ������һ���������е�һ��  a=z  
            end
        end
    end
    %  ���ɵ�һ��vector���棬���������~~  
    %  concatenate all end layer feature maps into vector
    net.fv = []; %%�����������һ�����ز�����Ӧ������ ��feature maps���ȫ���ӵ���ʽ 
    %%net.fv�� ���һ�����ز���������󣬲��õ���ȫ���ӷ�ʽ
    for j = 1 : numel(net.layers{n}.a) % fvÿ��ƴ����subFeaturemap2[j],[����50������]
        sa = size(net.layers{n}.a{j}); % ÿһ��featureMap�Ĵ�СΪa=sa=4*4*50,�õ�sfm2��һ������ͼ�Ĵ�С
        %rashape(A,m,n)
        %�Ѿ���A�ı���״����Ϊm��n�У�Ԫ�ظ������䣬ԭ�������ų�һ�ӣ��ٰ����ų����ɶӣ�
        %��net.layers{numLayers}.a[j](һ��sfm2)���г�Ϊ[4*4��,1��]��һ������
        %�����е�sfm2ƴ�ϳ�Ϊһ��������fv
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
        % ���õ�192*50�ľ���ÿһ�ж�Ӧһ������ͼ������� 
    end                        
    %feedforward into output perceptrons
    %net.ffW�ǡ�10,192����Ȩ�ؾ���
    %net.ffW*net.fv��һ����10,50���ľ���
    %repat(net.ffb,1,size(net.fv,2))��bias���Ƴ�50���ſ�
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2))); 
    %���Ϊ10*50�ľ���ÿһ�б�ʾһ������ͼ��ı�ǩ��� ȡ��sigmoid function������k����������������֮�䲻���⣬��ȻҲ���Ի���softmax�ع�  
end