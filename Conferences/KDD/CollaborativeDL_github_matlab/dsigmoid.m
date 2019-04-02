function [y] = dsigmoid(x, use_tanh)

if nargin < 2
    use_tanh = 0;
end

switch use_tanh
case 0
    y = x .* (1 - x);
case 1
    y = 1 - x.^2;
case 2
    y = x;
    y(x > 0) = 1;
    y(x <= 0) = 0;
end

