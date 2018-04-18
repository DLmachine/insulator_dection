function net = cnntrain(net, x, y, opts)  %%ѵ���Ĺ��̣�����bp�㷨����������  
    m = size(x, 4); %% mΪ������Ƭ��������size(x)=[28*28*6000] 
    numbatches = m / opts.batchsize;% ѭ���Ĵ��� ��1200�Σ�ÿ��ʹ��50����������
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL =0*y(1,1);%rL����С��������ƽ�����У���ͼҪ��
    for i = 1 : opts.numepochs    % ��������  
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %rnd_sel = randperm(size(x,3));
        %x=x(:,:,rnd_sel);
        %y=y(:,rnd_sel);
        tic;
        kk = randperm(m); %% �������m���ڵĲ��ظ���m����  
        for l = 1 : numbatches  %% ѭ��1200�Σ�ÿ��ѡȡ50�����ظ��������и��� 
            batch_x = x(:,:,:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%50��������ѵ������ 
            batch_y = y(:,kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%50����������Ӧ�ı�ǩ
            %batch_y=zeros(size(batch_y));
            net = cnnff(net, batch_x,opts);%����ǰ�򴫵����� 
            net = cnnbp(net, batch_y,opts);%���������򴫵��������ݶ�
            net = cnnapplygrads(net,opts); %% Ӧ���ݶȵ�������ģ��
            %net.LΪģ�͵�costFunction������С�������mse
            %rL����С��������ƽ������
            if isempty(net.rL)%Ϊ��
                net.rL(1) = net.L; %loss function��ֵ
            end
            net.rL(end + 1) = net.L; 
            %�൱�ڶ�ÿһ��batch���������ۻ�����Ȩƽ����
            disp(['cost ' num2str(l) '/' num2str(net.rL(end))]);
        end
        toc;
    end
end