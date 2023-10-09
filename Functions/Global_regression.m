function [bglobalr] = Global_regression(b)

% BOLD data analysis and visualization
%
% BOLD_plot(file,tr,ch_r,n_min,n_max,ix,iy,iz,ch_s,ch_t,n_fig)
%
% file         : root of the file name ('crs_srg' for 'crs_srg_no_regression_mask')
% tr           : trial number
% ch_r         : (0) no or (1) global regression
% n_min,n_max  : frame interval to analyze
% ix,iy,iz     : vectors of indices for the slice to analyze
% ch_s         : (1) seed method (0) nothing
% ch_t         : (1) temporal evolution (2) with movie (3) in real time (0) nothing 
% n_fig        : number of the 1st figure to plot
%
% ####### TBD: 
%               * SEED METHOD (ch_p = 1)
%                    - SEED ANYWHERE IN THE VOLUME (SEED COORD.?)
%                    - SIGNIFICANT LEVEL OF CORRELATION
%                    - COMPARISON WITHOUT/WITH GL. REGR.
%               * EXTRACT GLOBAL ARTIFACT WITH ICA?

[Nx,n_t] = size(b);

dt  = 1e-3;                 % (s)
t   = dt*(1:n_t)';

% Suppression of the temporal mean for each voxel

b_mt  = mean(b,2);
b_st  = std(b,0,2);
n_v   = sum(sum(b_st > 0));

% ##### SLICE global signal (with time dependent voxel series)

B_g = zeros(1,n_t);
v   = 0;
for n = 1:Nx
        b(n,:) = b(n,:) - b_mt(n).*ones(1,n_t);
        if b_st(n) > 0
            v   = v+1;
            B_g = B_g+squeeze(b(n,:));
        end
end

B_g  = B_g/n_v;
B_g2 = B_g*B_g';

gs = zeros(Nx,1);

for n = 1:Nx
        if b_st(n) > 0
            gs(n) = B_g*squeeze(b(n,:))'/B_g2;
        end
end

% Global signal regression

bglobalr=zeros(size(b));

for t = 1:n_t
    bglobalr(:,t) = b(:,t)-B_g(t)*gs;
end

