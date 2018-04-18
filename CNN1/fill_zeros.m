function rs = fill_zeros(input)
[h,w,c]=size(input);
rs=double(zeros([h*2+1,w*2+1,c]));
for i=1:c
   for j=1:h
    for k = 1:w
    rs(2*j,2*k,i)=input(j,k,i);
    end
   end
end
