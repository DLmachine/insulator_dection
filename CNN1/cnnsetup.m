function net = cnnsetup(net, x, y,opts)   %�Ը���������г�ʼ�� ����Ȩ�غ�ƫ��  
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 3;   %����ͼƬ����
    mapsize = size(squeeze(x(:, :, 1,1)));   % ͼƬ�Ĵ�С squeeze Ҫ��Ҫ����28 28,squeeze�Ĺ�����ɾ�������еĵ�һά

    for l = 1 : numel(net.layers)   %layer����
        if strcmp(net.layers{l}.type, 's')
            mapsize = (mapsize / net.layers{l}.scale);%% sumsampling��featuremap��������һ������featuremap�Ĳ��ص�ƽ��scale��С
            %�ֱ�Ϊ24/2=12;8/2=4
            %%assert:���Ժ���
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps % ������һ���ж���������ͼ��ͨ����ʼ��Ϊ1Ȼ��������µõ�
                net.layers{l}.b{j} = 0; 
                % ��ƫ�ó�ʼ��0, Ȩ��weight����δ���subsampling�㽫weight��Ϊ1/4 ��ƫ�ò�����Ϊ0����subsampling�׶��������  
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1; % �õ���ǰ��feature map�Ĵ�С�����ƽ�ƣ�Ĭ�ϲ���Ϊ1   
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;% ���ز�Ĵ�С����һ��(�������ͼ����)*(���������kernel�Ĵ�С)  
            for j = 1 : net.layers{l}.outputmaps  %��ǰ��feature maps�ĸ��� 
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;  %����ÿһ���������ͼ���ж��ٸ���������ǰ�㣬����Ȩ�ص������ֱ�Ϊ1*25��6*25
                for i = 1 : inputmaps  %  ����Ȩֵ����kernel��������Ϊinputmaps*outputmaps����,ÿһ����С����5*5  
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    if(opts.use_gpu==true)
                    net.layers{l}.k{i}{j}=gpuArray(net.layers{l}.k{i}{j}); end;
                    % ��ʼ��ÿ��feature map��Ӧ��kernel���� -0.5 �ٳ�2��һ����[-1,1]
                    % ���չ�һ����[-sqrt(6 / (fan_in + fan_out)),+sqrt(6 / (fan_in + fan_out))] why?? 
                end
                net.layers{l}.b{j} = 0; % ��ʼ��feture map��Ӧ��ƫ�ò��� ��ʼ��Ϊ0  
                if(opts.use_gpu==true)
                net.layers{l}.b{j}=gpuArray(net.layers{l}.b{j}); end;
            end
            inputmaps = net.layers{l}.outputmaps;% �޸�����feature maps�ĸ����Ա��´�ʹ��
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    %  onum �Ǳ�ǩ����Ҳ��������������Ԫ�ĸ�������Ҫ�ֶ��ٸ��࣬��Ȼ���ж��ٸ������Ԫ;
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    %  fvnum�����������ǰ��һ�����Ԫ����,��һ�����һ���Ǿ���pooling��Ĳ㣬
    %������inputmaps������map��ÿ������map�Ĵ�С��mapsize�����ԣ��ò����Ԫ������ inputmaps * ��ÿ������map�Ĵ�С��; 
    % 'ffb' is the biases of the output neurons.
    % ffb �������ÿ����Ԫ��Ӧ�Ļ�biases  
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    % ffW �����ǰһ�� �� ����� ���ӵ�Ȩֵ��������֮����ȫ���ӵ�  
    fvnum = prod(mapsize) * inputmaps;% prod��������������Ԫ�صĳ˻���fvnum=4*4*12=192,��ȫ���Ӳ���������� 
    onum = size(y, 1);%���շ���ĸ���  10��  ����������ڵ������
    % ���������һ����������趨 
    net.ffb = zeros(onum, 1);%softmat�ع��ƫ�ò������� 
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum)); %% softmaxt�ع��Ȩֵ���� Ϊ10*192�� ȫ����
end