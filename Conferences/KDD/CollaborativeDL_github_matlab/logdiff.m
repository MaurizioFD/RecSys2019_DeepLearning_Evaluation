%
% The original code was downloaed from 
% Ruslan Salakhutdinov\s hompeage:
%   http://http://www.mit.edu/~rsalakhu/
%
function ls = logdiff(xx,dim)
% ls = logsum(x,dim)
%
% returns the log of diff of logs
% similar to logsum.m function

if(length(xx(:))==1) ls=xx; return; end

xdims=size(xx);
if(nargin<2) 
  dim=find(xdims>1);
end

alpha = max(xx,[],dim)-log(realmax)/2;
repdims=ones(size(xdims)); repdims(dim)=xdims(dim);
ls = alpha+log(diff(exp(xx-repmat(alpha,repdims)),dim));

