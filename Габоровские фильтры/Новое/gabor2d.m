function im= gabor2d
f0 = 5;
u0 = f0;
P = pi;
x0 = 0;
a = 3.2;
teta = pi/2 + pi/6;

xx = -0.5:0.002:0.5;



f = zeros(length(xx),1);
f_ = zeros(length(xx),1);
g = zeros(length(xx),1);

for i = 1:length(xx)
        f(i) = cos(2*pi*(xx(i)*u0)+P);
        f_(i) = sin(2*pi*(xx(i)*u0)+P);
end;
figure, imshow(f, []);
figure, imshow(f_, []);

%%
% f = 0.5*(1+f);
% f_ = 0.5*(1+f_);

for i = 1:length(xx)

        rx =  (xx(i)-x0)*cos(teta);
        g(i) = exp(-pi * ( (a^2)*rx^2 ));
        if g(i) < 0.1
            g(i) = 0;
        end;

end;


im = f.*g;
im_ = f_.*g;

figure, plot(im);
figure, plot(im_);


