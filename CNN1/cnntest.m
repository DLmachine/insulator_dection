function [acc,h,net_r] = cnntest(net, x, y,opts)
    %  feedforward
    net1 = cnnff(net, x,opts);
    [~, h] = max(net1.o);
    [~, a] = max(y);
    bad = find(h ~= a);
    acc = 1-numel(bad) / size(y, 2);
    net_r=net1;
end
