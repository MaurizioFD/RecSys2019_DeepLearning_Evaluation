%
% The original code was downloaed from 
% Ruslan Salakhutdinov\s hompeage:
%   http://http://www.mit.edu/~rsalakhu/
%
function ls = logsum(xx,dim)
% ls = logsum(x,dim)
%
% returns the log of sum of logs
% computes ls = log(sum(exp(x),dim))
% but in a way that tries to avoid underflow/overflow
%
% basic idea: shift before exp and reshift back
% log(sum(exp(x))) = alpha + log(sum(exp(x-alpha)));
%
% This program was originally written by Sam Roweis

if(length(xx(:))==1) ls=xx; return; end

xdims=size(xx);
if(nargin<2) 
  dim=find(xdims>1);
end

alpha = max(xx,[],dim)-log(realmax)/2;
repdims=ones(size(xdims)); 
repdims(dim)=xdims(dim);
ls = alpha+log(sum(exp(xx-repmat(alpha,repdims)),dim));

