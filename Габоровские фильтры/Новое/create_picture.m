function img=create_picture
r0 = 0.5; %радиус кривизны
% центр кривизны
x0 = 1;
y0 = 0;
fi0=0;
P=pi;
x00 = x0 - r0*cos(fi0);
y00 = y0 - r0*sin(fi0);
f0=50;
xx = -0.5:0.001:0.5;
yy = -0.5:0.001:0.5;
f = zeros(length(xx),length(yy));
for i = 1:length(xx)
    for j = 1:length(yy)
        r = ((xx(i) - x00)^2 + (yy(j) - y00)^2)^0.5;
        img(i,j) = cos(2*pi*f0*(r)+P);
    end;
end;
img=img>0;
img=imresize(img,0.5);
img=double(img);
figure, imshow(img, []);