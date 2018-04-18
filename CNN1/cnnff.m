function net = cnnff(net, x,opts)%完成训练的前向过程，  
    n = numel(net.layers);
    net.layers{1}.a{1} =reshape(x(:,:,1,:),[size(x,1),size(x,2),size(x,4)]); % a是输入map,是一个【128,128,50】的矩阵
    net.layers{1}.a{2} =reshape(x(:,:,2,:),[size(x,1),size(x,2),size(x,4)]);
    net.layers{1}.a{3} =reshape(x(:,:,3,:),[size(x,1),size(x,2),size(x,4)]); 
    inputmaps = 3;
    for l = 2 : n   %针对每一个卷积层
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps   % 针对该层的每一个feture map  
                %create temp output map
                %z = zeros(size(net.layers{l-1}.a{1}));
                z = zeros(size(net.layers{l - 1}.a{1})-[net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                if(opts.use_gpu==true) 
                z=gpuArray(z);end;
                % 该层feture map的大小，最后一位是样本图片个数 初始化为0  
                for i = 1 : inputmaps  %针对每一个输入feature map
                    z = z + convn(net.layers{l-1}.a{i}, net.layers{l}.k{i}{j}, 'valid');    
                    %做卷积操作  k{i}{j} 是5*5的double类型，其中a{i}是输入图片的feature map 大小为28*28*50 50为图像数量  
                    %卷积操作这里的k已经旋转了180度，z保存的就是该层中所有的特征
                end
                % 通过非线性加偏值
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});%%获取sigmoid function的值 
            end
            inputmaps = net.layers{l}.outputmaps; %% 设置新的输入feature maps的个数      
        elseif strcmp(net.layers{l}.type, 's') %下采样采用的方法是,2*2相加乘以权值1/4,  没有取偏置和取sigmoid  
            %  downsample
            for j = 1 : inputmaps
                %这里做的是mean-pooling
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');  
                 %先卷积后各行各列取结果
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                 %得到的结果是上一层卷积层行列的一半  a=z  
            end
        end
    end
    %  收纳到一个vector里面，方便后面用~~  
    %  concatenate all end layer feature maps into vector
    net.fv = []; %%用来保存最后一个隐藏层所对应的特征 将feature maps变成全连接的形式 
    %%net.fv： 最后一层隐藏层的特征矩阵，采用的是全连接方式
    for j = 1 : numel(net.layers{n}.a) % fv每次拼合入subFeaturemap2[j],[包含50个样本]
        sa = size(net.layers{n}.a{j}); % 每一个featureMap的大小为a=sa=4*4*50,得到sfm2的一个输入图的大小
        %rashape(A,m,n)
        %把矩阵A改变形状，变为m行n列（元素个数不变，原矩阵按列排成一队，再按行排成若干队）
        %把net.layers{numLayers}.a[j](一个sfm2)排列成为[4*4行,1列]的一个向量
        %把所有的sfm2拼合成为一个列向量fv
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
        % 最后得到192*50的矩阵，每一列对应一个样本图像的特征 
    end                        
    %feedforward into output perceptrons
    %net.ffW是【10,192】的权重矩阵
    %net.ffW*net.fv是一个【10,50】的矩阵
    %repat(net.ffb,1,size(net.fv,2))把bias复制成50份排开
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2))); 
    %结果为10*50的矩阵，每一列表示一个样本图像的标签结果 取了sigmoid function表明是k个二分类器，各类之间不互斥，当然也可以换成softmax回归  
end