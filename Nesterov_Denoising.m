%% This program is used for image denoising
% using total variation (TV) and Nesterov's first order method

%-------------Description------------------
% min ||x||_TV s.t. ||x-b||_l2 < delta

% b= original noisy image
% x= reconstructed denoised image
% delta = threshold error

% Ref: Dahl et al.,"Algorithms and software for 
% total variation image reconstruction via first-order 
%methods" Numer. Algor.,53: 67-92 (2010)
%-------------------------------------------
% Author: Sovan Mukherjee, April, 2015

clear all;close all; clc;

%% Shepp-Logan 

N=1024;                                                                    %Pixel size of the image
Im_original= phantom(N);                                                   %original Shepp-Logan Phantom

% Add some gaussian noise to the image

sigma= 0.10;                                                               %standard deviation of the noise
b = Im_original+sigma*randn(size(Im_original,1)); % noisy image

figure;

imshow(Im_original); 

title('Original Image');

figure;

imshow(b);                                                                 % show the noisy image

title('Noisy Image');

%% ------------------------------------------------------------------------

%% -- implement Nesterov's first order method -----------------------------
x= b;                                                                      % initialize denoised image as the original noisy image
% set up the parameters
m= size(Im_original,1);                                                    % no. of image pixels in x-direction
n= size(Im_original,2);                                                    % no. of image pixels in y-direction

x= reshape(x, m*n,1);                                                      % tranform the image matrix into a vector
b= reshape(b, m*n,1);


e_rel =5e-4;                                                               % relative epsilon

% epsilon= norm(b, Inf)*m*n*e_rel;                                         % thresold parameter for stopping the iteration

epsilon= max(b(:))*m*n*e_rel;                                              % thresold parameter for stopping the iteration

mu= epsilon/(m*n);                                                         % smoothing parameter
tao=0.99;
delta= tao*sqrt(m*n)*sigma;                                                % threshold relative error between noisy and denoised image

% find the gradient operator

[D1,D2]= grad_operator_new(size(Im_original,1));                           % calculating gradient operator

D= [D1;D2];                                                                % gradient operator

L_mu= 8/mu;                                                                % Lipschitz constant of the gradient

% Nesterov's algorithm starts here
max_iter=300;                                                              % maximum number of iteration
wk= zeros(m*n,1);

tic;

for k =0:max_iter-1
    
    fprintf('iteration # %d\t',k); 
    
    alphak= 0.5*(k+1);
    
    Dhx= D1*x;
    Dvx= D2*x;
    
    D_x= [Dhx;Dvx];                                                        % gradient of x
        
    % compute gk (step 1)
    w=max(mu,sqrt(abs(Dhx).^2+abs(Dvx).^2));
    u1=Dhx./w;
    u2=Dvx./w;
    uk=[u1;u2];
    
    gk= D'*uk;                                                           
    
    % compute yk (step 2)
    yk1= L_mu*(x-b)-gk;
    
%   yk1= -L_mu*(x-b)+gk;

    yk2= max(L_mu,norm(yk1,2)/delta);
    
    yk = yk1/yk2+b;
    
    % compute zk (step 3)
    
    wk= wk+alphak*gk;
    
    zk1= -wk;
    
    zk2= max(L_mu, norm(wk,2)/delta);
    
    zk= zk1/zk2+b;
    
    % update xk (step 4)
    
    x= (2/(k+3))*zk+((k+1)/(k+3))*yk;
    
    %stopping criterion
    
    D_x_norm= sum(sqrt(abs(Dhx).^2+abs(Dvx).^2));
    
%     D_x_sum= sum(D*x);
    
    stop_parameter= D_x_norm+delta*norm(gk,2)-uk'*D*b;                     %stopping parameter
    
    if stop_parameter<epsilon
        
        break;
        
    end
    
    fprintf('Relative error= %f\n',stop_parameter);
       
end

t.tv=toc;

fprintf('\nReconstruction time (s)= %f\n',t.tv);
%% ----- Nesterov's algorithm ends here -----------------------------------

%% show the denoised reconstructed image

x= reshape(x, size(Im_original,1),size(Im_original,2));

b= reshape(b, size(Im_original,1),size(Im_original,2));

figure;

imshow(x); % show the denoised image 

title('denoised image');

%% Calculate peak signal-to-noise ratio (PSNR)

[m_0,n_0]= size(Im_original);

% PSNR for noisy image

mse_n= (1/(m_0*n_0))*sum(sum((Im_original-b).^2));

max= max(Im_original(:));

PSNR_dB_noised= 10*log10(max^2/mse_n);

% PSNR for denoised image

mse= (1/(m_0*n_0))*sum(sum((Im_original-x).^2));

PSNR_dB_denoised= 10*log10(max^2/mse);

