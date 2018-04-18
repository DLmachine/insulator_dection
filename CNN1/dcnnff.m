function r_net = dcnnff(net,s_layer,s_map)%
    n = numel(net.layers);  
    m=1;
    for l=n:-1:2
      if strcmp(net.layers{l}.type, 'c')  
        r_net.layers{m}=net.layers{l};
        r_net.layers{m}.outputmaps=numel(net.layers{l-1}.a);  
        m=m+1;
       end
    end
    r_net.layers{m}=net.layers{1};
    r_net.layers{m+1}=net.layers{1};
    %s_layer=n+1-s_layer;
    inputmaps = 1;
    for l = s_layer:1:numel(r_net.layers)    %
        if strcmp(r_net.layers{l}.type, 'c')
            for j = 1 : r_net.layers{l}.outputmaps   % 
                %create temp output map
                z = zeros(size(r_net.layers{l+1}.a{1}));
                for i = 1 : inputmaps  %feature map
                    z = z+imresize(convn(fill_zeros(r_net.layers{l}.a{i+s_map-1}),(r_net.layers{l}.k{j}{i+s_map-1})','valid'),'OutputSize',size(z(:,:,1)));    
                end
                %r_net.layers{l+1}.a{j} = sigm(z - r_net.layers{l}.b{j});
                r_net.layers{l+1}.a{j} = z/inputmaps;
            end
         inputmaps = r_net.layers{l}.outputmaps; %%
         s_map=1;  
        %{
        elseif strcmp(r_net.layers{l}.type, 's')% pooling layer
            % downsample
            for j = 1 : inputmaps
                % mean-pooling
                r_net.layers{l}.a{j+s_map-1}=imresize(r_net.layers{l}.a{j+s_map-1},'OutputSize',[2*size(r_net.layers{l}.a{j+s_map-1},1),2*size(r_net.layers{l}.a{j+s_map-1},2)]);
                %net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                %z = convn(net.layers{l+1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');  
            end
        s_map=1;
        %}
        end  
    end
end
