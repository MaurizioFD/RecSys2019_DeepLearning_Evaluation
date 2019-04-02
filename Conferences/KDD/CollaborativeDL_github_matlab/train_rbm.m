% rbm - training restricted Boltzmann machine using Gibbs sampling
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [R_out] = train_rbm (R, patches);

    n_visible = R.structure.n_visible;
    n_hidden = R.structure.n_hidden;

    if length(R.vbias) ~= n_visible
        warning ('Reinitializing visible biases');
        R.vbias_init = zeros(n_visible, 1);
        R.vbias = R.vbias_init;
    end

    if length(R.hbias) ~= n_hidden
        warning ('Reinitializing hidden biases');
        R.hbias_init = zeros(n_hidden, 1);
        R.hbias = R.hbias_init;
    end

    if sum( (size(R.W) - [n_visible n_hidden]).^2 ) ~= 0
        warning ('Reinitializing weights');
        R.W_init = R.learning.weight_scale * 2 * (rand(n_visible, n_hidden) - 0.5);
        R.W = R.W_init;
    end

    % TODO: Merge rbm_pt.m and grbm_pt.m
    if R.parallel_tempering.use == 1
        if R.data.binary == 1
            R_out = rbm_pt(R, patches);
        else
            R_out = grbm_pt(R, patches);
        end
    else
        R_out = rbm(R, patches);
    end

