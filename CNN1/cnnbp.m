function net = cnnbp(net, y,opts)%���㲢�����������error���������ݶȣ�Ȩ�ص��޸����� 
    n = numel(net.layers); %layers����� 
    %   error
    net.e = net.o - y; % 10*50  ÿһ�б�ʾһ������ͼ��
    %  loss function���������
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%û�м���������ɱ�Ҷ˹ѧ�ɵĹ۵�

    %% backprop deltas
    %����β�������֪�������
    net.od = net.e .* (net.o .* (1 - net.o));   %  ��������� sigmoid���
    %fvd,feature vector delta,������������һ���ռ��²���size=[192*50]
    net.fvd = (net.ffW' * net.od);            
    %���MLP��ǰһ�㣨������ȡ���һ�㣩�Ǿ���㣬�������������sigmoid��������error��Ҫ��
    %������ʶ����������в���Ҫ�õ�
    if strcmp(net.layers{n}.type, 'c')          %  only conv layers has sigm function
        %���ھ���㣬��������������� ����
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv)); %% ������һ�����ز��Ǿ���㣬ֱ���øù�ʽ���ܵõ����
    end

    %  reshape feature vector deltas into output map style
    %�ѵ����֪���������featureVector �������󣬻ָ�ΪsubFeatureMap2��4*4��λ������ʽ
    sa = size(net.layers{n}.a{1}); %size��a{1}��=[4*4*50],һ����a{1}~a{12}}
    fvnum = sa(1) * sa(2);%fvnum һ��ͼ�����е���������������4*4
    for j = 1 : numel(net.layers{n}.a)   %subFeatureMap��������1:12
        %net���һ���delta������������delta������ȡһ��featureMap��С��Ȼ����ϳ�Ϊһ��featureMap����״
        %��fvd���汣���������������������������cnnff.m������������map���ɵģ���������Ҫ���±任��������map����ʽ
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
        %size(net.layers{numLayers}.d{j})=��4*4*50��
        %size��net.fvd��=[192*50]
    end

    for l = (n - 1) : -1 : 1   %ʵ���ǵ�2��ֹ�ˣ�1������㣬û�����Ҫ��
        %l���Ǿ���㣬�����²㣨�������㴫���������ôӺ���ǰ��̯�ķ�ʽ�������ϲ������̯2�����ٳ���4
        if strcmp(net.layers{l}.type, 'c')  %�����ļ��㷽ʽ
            for j = 1 : numel(net.layers{l}.a)   %��n-1����е�feature maps�ĸ��������б��� ÿ��d{j}��8*8*50����ʽ�� ������һ��Ϊ�²����㣬�ʺ�һ��d{j}��չΪ8*8�ģ�ÿ���㸴�Ƴ�2*2�ģ�,����bp����ʽ�Ϳ��Եó�������Ȩ�ؾ�Ϊ1/4,
                %net.layers{l}.a{j}.*(1-net.layers{l}.a{j})Ϊsigmoid�ĵ���
                %(expand(net.layers{l+1}.d{j},[net.layers{j+1}.scale net.layers{l+1}.scale 1]))
                %expand����ʽչ�����
                %net.layers{l+1}.scale^2
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);%�����ڿ˻��������䣬��������ľ����Ĳвd��ʾ�вa��ʾ���ֵ
            end
            %l���ǽ������㣬�����²㣨����㴫���������þ���ķ�ʽ�õ�
        elseif strcmp(net.layers{l}.type, 's') 
            for i = 1 : numel(net.layers{l}.a)  %l������������
                z = zeros(size(net.layers{l}.a{1}));  %z�õ�һ��feature map�Ĵ�С�������
                if(opts.use_gpu==true) 
                z=gpuArray(z);end;
                for j = 1 : numel(net.layers{l + 1}.a) %��l+1���ռ�����
                    %net.layers{l+1}.d{j}��һ�㣨����㣩��������
                    %net.layers{l+1}.k{i}{j},��һ�㣨����㣩�ľ����
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end  
                net.layers{l}.d{i} = z;  %% ��Ϊ���²����㣬����a=z,��f(z)=z,�����͵���1�����������������ӽ��Ȩֵ���һ������  
            end
        end
    end

    %%  calc gradients    
    %����������ȡ�㣨���+�����������ݶ�
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')%�����
            for j = 1 : numel(net.layers{l}.a)%l���featureMap������
                for i = 1 : numel(net.layers{l - 1}.a)
                    %����˵��޸���=����ͼ��*���ͼ���delta
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);% ���Կ������е��Ƶ������������Ƚ�k rot180��Ȼ����rot����Ч����һ���ġ� 
                end
                %net.layers{l}.d{j}(:)��һ��24*24*50�ľ���db������50
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);  %% ��ƫ�ò���b�ĵ��� 
            end
        end
    end
    %����β�������֪�����ݶ�
    %sizeof(net.od)=[10,50]
    %�޸�������ͳ���50��batch��С��
    net.dffW = net.od * (net.fv)' / size(net.od, 2);  %softmax�ع��в�������Ӧ�ĵ���  
    net.dffb = mean(net.od, 2);%% �ڶ�άȡ��ֵ  

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);% flipdim(X, 1) �л���  flipdim(X, 2) �л���  
    end
end