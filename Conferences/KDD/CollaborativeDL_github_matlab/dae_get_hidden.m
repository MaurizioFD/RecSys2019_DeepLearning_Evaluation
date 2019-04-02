% dae_get_hidden
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [h] = dae_get_hidden(x0, D, target_sparsity)

if nargin < 3
    target_sparsity = 0;
end

h = bsxfun(@plus, x0 * D.W, D.hbias');

if D.hidden.binary == 1
    h = sigmoid(h, D.hidden.use_tanh);

    if target_sparsity > 0 && D.hidden.use_tanh == 0
        avg_acts = mean(h, 1);
        diff_acts = max(avg_acts - (1 - target_sparsity), 0);
        h = min(max(bsxfun(@minus, h, diff_acts), 0), 1);
    end
end

