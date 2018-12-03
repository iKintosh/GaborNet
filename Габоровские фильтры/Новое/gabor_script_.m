clear all;
close all;

f0 = 5;
fi0 = pi + pi/6; %угол наклона касательной к синусоиде в центре гауссова окна
u0 = f0*cos(fi0);
v0 = f0*sin(fi0);
P = pi;
x0 = 0;
y0 = 0;
a = 3.2;
b = 4.8;
teta = pi/2 + pi/6;

r0 = 0.5; %радиус кривизны
% центр кривизны
x00 = x0 - r0*cos(fi0);
y00 = y0 - r0*sin(fi0);

xx = -0.5:0.002:0.5;
yy = -0.5:0.002:0.5;



f = zeros(length(xx),length(yy));
f_ = zeros(length(xx),length(yy));
fr = zeros(length(xx),length(yy));
fr_ = zeros(length(xx),length(yy));
g = zeros(length(xx),length(yy));

for i = 1:length(xx)
    for j = 1:length(yy)
        r = ((xx(i) - x00)^2 + (yy(j) - y00)^2)^0.5;
        fr(i,j) = cos(2*pi*f0*(r)+P);
        fr_(i,j) = sin(2*pi*f0*(r)+P);
        f(i,j) = cos(2*pi*(xx(i)*u0+yy(j)*v0)+P);
        f_(i,j) = sin(2*pi*(xx(i)*u0+yy(j)*v0)+P);
    end;
end;
figure, imshow(f, []);
figure, imshow(f_, []);
figure, imshow(fr, []);
figure, imshow(fr_, []);

%%
% f = 0.5*(1+f);
% f_ = 0.5*(1+f_);

for i = 1:length(xx)
    for j = 1:length(yy)
        rx =  (xx(i)-x0)*cos(teta) + (yy(j)-x0)*sin(teta);
        ry = -(xx(i)-x0)*sin(teta) + (yy(j)-x0)*cos(teta);
        g(i,j) = exp(-pi * ( (a^2)*rx^2 + (b^2)*ry^2) );
        if g(i,j) < 0.1
            g(i,j) = 0;
        end;
    end;
end;


im = f.*g;
im_ = f_.*g;
imr = fr.*g;
imr_ = fr_.*g;

figure, imshow(im, []);
figure, imshow(im_, []);
figure, imshow(imr, []);
figure, imshow(imr_, []);


