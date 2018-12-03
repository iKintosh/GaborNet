function img1=add_noise(img,isblur)
sigma=10;
ft=fft(img);
ft=fftshift(ft);
ft=ft+random('norm', 0, sigma,size(img,1), size(img,2));
ft=ifftshift(ft);
img1=ifft(ft);
img1=mat2gray(abs(img1));

if isblur;
    filter=fspecial('gaussian',[7 7], 3);
    img1 = imfilter(img1, filter);
    img1=mat2gray(img1);
    
    sigma=10;
    ft=fft(img1);
    ft=fftshift(ft);
    ft=ft+random('norm', 0, sigma,size(img,1), size(img,2));
    ft=ifftshift(ft);
    img1=ifft(ft);
    img1=mat2gray(abs(img1));
end;

