function net = cnnsetup(net, x, y,opts)   %对各层参数进行初始化 包括权重和偏置  
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 3;   %输入图片数量
    mapsize = size(squeeze(x(:, :, 1,1)));   % 图片的大小 squeeze 要不要都行28 28,squeeze的功能是删除矩阵中的单一维

    for l = 1 : numel(net.layers)   %layer层数
        if strcmp(net.layers{l}.type, 's')
            mapsize = (mapsize / net.layers{l}.scale);%% sumsampling的featuremap长宽都是上一层卷积层featuremap的不重叠平移scale大小
            %分别为24/2=12;8/2=4
            %%assert:断言函数
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps % 就是上一层有多少张特征图，通过初始化为1然后依层更新得到
                net.layers{l}.b{j} = 0; 
                % 将偏置初始化0, 权重weight，这段代码subsampling层将weight设为1/4 而偏置参数设为0，故subsampling阶段无需参数  
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1; % 得到当前层feature map的大小，卷积平移，默认步长为1   
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;% 隐藏层的大小，是一个(后层特征图数量)*(用来卷积的kernel的大小)  
            for j = 1 : net.layers{l}.outputmaps  %当前层feature maps的个数 
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;  %对于每一个后层特征图，有多少个参数链到前层，包含权重的总数分别为1*25；6*25
                for i = 1 : inputmaps  %  共享权值，故kernel参数个数为inputmaps*outputmaps个数,每一个大小都是5*5  
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    if(opts.use_gpu==true)
                    net.layers{l}.k{i}{j}=gpuArray(net.layers{l}.k{i}{j}); end;
                    % 初始化每个feature map对应的kernel参数 -0.5 再乘2归一化到[-1,1]
                    % 最终归一化到[-sqrt(6 / (fan_in + fan_out)),+sqrt(6 / (fan_in + fan_out))] why?? 
                end
                net.layers{l}.b{j} = 0; % 初始话feture map对应的偏置参数 初始化为0  
                if(opts.use_gpu==true)
                net.layers{l}.b{j}=gpuArray(net.layers{l}.b{j}); end;
            end
            inputmaps = net.layers{l}.outputmaps;% 修改输入feature maps的个数以便下次使用
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    %  onum 是标签数，也就是最后输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元;
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    %  fvnum是最后输出层的前面一层的神经元个数,这一层的上一层是经过pooling后的层，
    %包含有inputmaps个特征map。每个特征map的大小是mapsize。所以，该层的神经元个数是 inputmaps * （每个特征map的大小）; 
    % 'ffb' is the biases of the output neurons.
    % ffb 是输出层每个神经元对应的基biases  
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    % ffW 输出层前一层 与 输出层 连接的权值，这两层之间是全连接的  
    fvnum = prod(mapsize) * inputmaps;% prod函数用于求数组元素的乘积，fvnum=4*4*12=192,是全连接层的输入数量 
    onum = size(y, 1);%最终分类的个数  10类  ，最终输出节点的数量
    % 这里是最后一层神经网络的设定 
    net.ffb = zeros(onum, 1);%softmat回归的偏置参数个数 
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum)); %% softmaxt回归的权值参数 为10*192个 全连接
end