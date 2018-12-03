f0 = 5;
s=2;
fi0 = pi + pi/6; %угол наклона касательной к синусоиде в центре гауссова окна
u0 = f0*cos(fi0);
v0 = f0*sin(fi0);
x0 = 0;
y0 = 0;
a = 3.2;
b = 4.8;

% центр кривизны
r0=1;
x00 = x0 - r0*cos(fi0);
y00 = y0 - r0*sin(fi0);

xx = -0.5:0.002:0.5;
yy = -0.5:0.002:0.5;

f_ = zeros(length(xx),length(yy));
g = zeros(length(xx),length(yy));
for k = 1:length(xx)
    for j = 1:length(yy)
        %f_(i,j) = sin(2*pi*(xx(i)*u0+yy(j)*v0));
        f_x(k)=1;%cos(2*pi*u0*xx(k))+1i*sin(2*pi*u0*xx(k));
        f_y(j)=(cos(2*pi*v0*yy(j)))^2;%+1i*sin(2*pi*v0*yy(j));
        f_(k,j)=real((f_x(k)*f_y(j)));
    end;
end;
figure, imshow(f_, []);


for k = 1:length(xx)
    for j = 1:length(yy)
        rx =  (xx(k));
        ry =  (yy(j));
        g(k,j) = exp(-pi * ( (a^2)*rx^2 + (b^2)*ry^2) );
        if g(k,j) < 0.1
            g(k,j) = 0;
        end;
    end;
end;


im_ = f_.*g;

figure, imshow(im_, []);
