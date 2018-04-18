function net = cnntrain(net, x, y, opts)  %%训练的过程，包括bp算法及迭代过程  
    m = size(x, 4); %% m为样本照片的数量，size(x)=[28*28*6000] 
    numbatches = m / opts.batchsize;% 循环的次数 共1200次，每次使用50个样本进行
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL =0*y(1,1);%rL是最小均方误差的平滑序列，绘图要用
    for i = 1 : opts.numepochs    % 迭代次数  
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %rnd_sel = randperm(size(x,3));
        %x=x(:,:,rnd_sel);
        %y=y(:,rnd_sel);
        tic;
        kk = randperm(m); %% 随机产生m以内的不重复的m个数  
        for l = 1 : numbatches  %% 循环1200次，每次选取50个不重复样本进行更新 
            batch_x = x(:,:,:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%50个样本的训练数据 
            batch_y = y(:,kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%50个样本所对应的标签
            %batch_y=zeros(size(batch_y));
            net = cnnff(net, batch_x,opts);%计算前向传导过程 
            net = cnnbp(net, batch_y,opts);%计算误差并反向传导，计算梯度
            net = cnnapplygrads(net,opts); %% 应用梯度迭代更新模型
            %net.L为模型的costFunction，即最小均方误差mse
            %rL是最小均方误差的平滑序列
            if isempty(net.rL)%为空
                net.rL(1) = net.L; %loss function的值
            end
            net.rL(end + 1) = net.L; 
            %相当于对每一个batch的误差进行累积（加权平均）
            disp(['cost ' num2str(l) '/' num2str(net.rL(end))]);
        end
        toc;
    end
end