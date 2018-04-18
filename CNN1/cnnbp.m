function net = cnnbp(net, y,opts)%计算并传递神经网络的error，并计算梯度（权重的修改量） 
    n = numel(net.layers); %layers层个数 
    %   error
    net.e = net.o - y; % 10*50  每一列表示一个样本图像
    %  loss function，均方误差
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%没有加入参数构成贝叶斯学派的观点

    %% backprop deltas
    %计算尾部单层感知机的误差
    net.od = net.e .* (net.o .* (1 - net.o));   %  输出层的误差 sigmoid误差
    %fvd,feature vector delta,特征向量误差，上一层收集下层误差，size=[192*50]
    net.fvd = (net.ffW' * net.od);            
    %如果MLP的前一层（特征抽取最后一层）是卷基层，卷基层的输出经过sigmoid函数处理，error需要求导
    %在数字识别这个网络中不需要用到
    if strcmp(net.layers{n}.type, 'c')          %  only conv layers has sigm function
        %对于卷基层，它的特征向量误差 再求导
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv)); %% 如果最后一个隐藏层是卷积层，直接用该公式就能得到误差
    end

    %  reshape feature vector deltas into output map style
    %把单层感知机的输入层featureVector 的误差矩阵，恢复为subFeatureMap2的4*4二位矩阵形式
    sa = size(net.layers{n}.a{1}); %size（a{1}）=[4*4*50],一共有a{1}~a{12}}
    fvnum = sa(1) * sa(2);%fvnum 一个图所含有的特征向量数量，4*4
    for j = 1 : numel(net.layers{n}.a)   %subFeatureMap的数量，1:12
        %net最后一层的delta，由特征向量delta，依次取一个featureMap大小，然后组合成为一个featureMap的形状
        %在fvd里面保存的是所有样本的特征向量（在cnnff.m函数中用特征map拉成的），这里需要重新变换回来特征map的形式
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
        %size(net.layers{numLayers}.d{j})=【4*4*50】
        %size（net.fvd）=[192*50]
    end

    for l = (n - 1) : -1 : 1   %实际是到2终止了，1是输入层，没有误差要求
        %l层是卷积层，误差从下层（降采样层传来），采用从后往前均摊的方式传播误差，上层误差内摊2倍，再除以4
        if strcmp(net.layers{l}.type, 'c')  %卷积层的计算方式
            for j = 1 : numel(net.layers{l}.a)   %第n-1层具有的feature maps的个数，进行遍历 每个d{j}是8*8*50的形式， 由于下一层为下采样层，故后一层d{j}扩展为8*8的（每个点复制成2*2的）,按照bp求误差公式就可以得出，这里权重就为1/4,
                %net.layers{l}.a{j}.*(1-net.layers{l}.a{j})为sigmoid的倒数
                %(expand(net.layers{l+1}.d{j},[net.layers{j+1}.scale net.layers{l+1}.scale 1]))
                %expand多项式展开相乘
                %net.layers{l+1}.scale^2
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);%克罗内克积进行扩充，求采样曾的卷基层的残差，d表示残差，a表示输出值
            end
            %l层是降采样层，误差从下层（卷积层传来），采用卷积的方式得到
        elseif strcmp(net.layers{l}.type, 's') 
            for i = 1 : numel(net.layers{l}.a)  %l层输出层的数量
                z = zeros(size(net.layers{l}.a{1}));  %z得到一个feature map的大小的零矩阵
                if(opts.use_gpu==true) 
                z=gpuArray(z);end;
                for j = 1 : numel(net.layers{l + 1}.a) %从l+1层收集错误
                    %net.layers{l+1}.d{j}下一层（卷积层）的灵敏度
                    %net.layers{l+1}.k{i}{j},下一层（卷积层）的卷积核
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end  
                net.layers{l}.d{i} = z;  %% 因为是下采样层，所以a=z,就f(z)=z,导数就等于1，所以误差就是所连接结点权值与后一层误差和  
            end
        end
    end

    %%  calc gradients    
    %计算特征抽取层（卷积+降采样）的梯度
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')%卷积层
            for j = 1 : numel(net.layers{l}.a)%l层的featureMap的数量
                for i = 1 : numel(net.layers{l - 1}.a)
                    %卷积核的修改量=输入图像*输出图像的delta
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);% 可以看论文中的推导！与论文中先将k rot180，然后再rot整体效果是一样的。 
                end
                %net.layers{l}.d{j}(:)是一个24*24*50的矩阵，db仅除以50
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);  %% 对偏置参数b的导数 
            end
        end
    end
    %计算尾部单层感知机的梯度
    %sizeof(net.od)=[10,50]
    %修改量，求和除以50（batch大小）
    net.dffW = net.od * (net.fv)' / size(net.od, 2);  %softmax回归中参数所对应的导数  
    net.dffb = mean(net.od, 2);%% 第二维取均值  

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);% flipdim(X, 1) 行互换  flipdim(X, 2) 列互换  
    end
end