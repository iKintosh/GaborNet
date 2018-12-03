xx = -0.5:0.002:0.5;
yy = -0.5:0.002:0.5;
f0=5;

for i = 1:length(xx)
    fx(i)=2*pi*f0*xx(i);
    fy(i)=2*pi*f0*yy(i);
end;

for i=1:length(xx);
    for j=1:length(yy);
        f(i,j)=cos(fx(i))*cos(fy(j));
    end;
end;
figure; imshow(f, []);