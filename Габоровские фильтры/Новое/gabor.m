function filt=gabor
sigma=2*pi;
kmax=pi/2;
freq=sqrt(2);
scales=[0 1 2 3 4];
orientations=[0 1 2 3 4 5 6 7];

xx = -30:0.2:30;
yy = -30:0.2:30;

for v=1:length(scales);
    for mu=1:length(orientations);
        fi=pi*mu/8;
        kv=kmax/(freq^v);
        k=kv*exp(1i*fi);
        disp(['calculating at frequency: ' mat2str(v) ' orientation: ' mat2str(mu)]);
        f = zeros(length(xx),length(yy));
        g = zeros(length(xx),length(yy));
        for i = 1:length(xx)
            for j = 1:length(yy)
               f(i,j)=(exp(1i*(kv*xx(i)*cos(fi)+kv*yy(j)*sin(fi)))-exp(-sigma^2/2))*(norm(k))^2/(sigma^2);
               g(i,j)=exp(-(norm(k))^2*(xx(i)^2+yy(j)^2)/(2*sigma^2));
            end;
            filt(v,mu)={f.*g};
        end;
    end;
end;
k=1;
for i=1:length(scales);
    for j=1:length(orientations);
       subplot(length(scales),length(orientations),k); imshow(real(filt{i,j}), []);
       k=k+1;
    end;
end;