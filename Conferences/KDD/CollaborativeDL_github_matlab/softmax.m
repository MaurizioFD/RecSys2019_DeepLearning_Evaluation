% simple wrapper for softmax function
function [y] = softmax(x)

logZ = logsum(x, 2);
y = exp(bsxfun(@minus, x, logZ));


