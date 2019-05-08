% This function calculates gradient operator

% Author: Sovan Mukherjee, August, 2014

function [del_x,del_y]=grad_operator_new(N)
 
% gradient operator
D= spdiags([-ones(N,1) ones(N,1)],[0,1],N,N);

D(N,:)=0;

del_x = kron(speye(N),D);                                                  % gradient matrix in horizontal direction

del_y= kron(D, speye(N));                                                  % gradient matrix in vertical direction

