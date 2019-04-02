% default_dae - 
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
function [D] = default_dae (n_visible, n_hidden);
    % data type
    D.data.binary = 0;
    %D.data.binary = 1; 

    % hidden type: sigmoid or linear
    D.hidden.binary = 1;
    %D.data.binary = 0; % for linear units

    % nonlinearity: the name of the variable will change in the later revision
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    D.hidden.use_tanh = 0;

    % use sigmoid nonlinearity
    D.visible.use_tanh = 0;
    %% use tanh nonlinearity 
    %D.visible.use_tanh = 1;

    % learning parameters
    D.learning.lrate = 1e-2;
    D.learning.lrate0 = 5000;
    D.learning.momentum = 0;
    D.learning.weight_decay = 0;
    D.learning.weight_scale = sqrt(6)/sqrt(n_visible + n_hidden);
    D.learning.minibatch_sz = 100;
    D.learning.lrate_anneal = 0.9;

    D.valid_min_epochs = 10;

    % sparsity
    D.sparsity.target = 0;
    D.sparsity.cost = 0;

    % cae
    D.cae.cost = 0;

    % Gaussian-Bernoulli RBM
    D.do_normalize = 1;
    D.do_normalize_std = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    D.stop.criterion = 0;
    % criterion == 1
    D.stop.recon_error.tolerate_count = 1000;

    % denoising
    D.noise.drop = 0.1;
    D.noise.level = 0.1;

    % structure
    D.structure.n_visible = n_visible;
    D.structure.n_hidden = n_hidden;

    % reconstruct ICA
    D.rica.cost = 0;

    % initializations
    D.W_init = 2 * D.learning.weight_scale * (rand(n_visible, n_hidden) - 0.5);
    D.vbias_init = zeros(n_visible, 1);
    D.hbias_init = zeros(n_hidden, 1);
    D.W = D.W_init;
    D.vbias = D.vbias_init;
    D.hbias = D.hbias_init;

    % adagrad
    D.adagrad.use = 0;
    D.adagrad.epsilon = 1e-8;
    D.adagrad.W = zeros(n_visible, n_hidden);
    D.adagrad.vbias = zeros(n_visible, 1);
    D.adagrad.hbias = zeros(n_hidden, 1);

    % adadelta
    D.adadelta.use = 0;
    D.adadelta.epsilon = 1e-6;
    D.adadelta.momentum = 0.99;
    D.adadelta.gW = zeros(n_visible, n_hidden);
    D.adadelta.gvbias = zeros(n_visible, 1);
    D.adadelta.ghbias = zeros(n_hidden, 1);
    D.adadelta.W = zeros(n_visible, n_hidden);
    D.adadelta.vbias = zeros(n_visible, 1);
    D.adadelta.hbias = zeros(n_hidden, 1);

    % iteration
    D.iteration.n_epochs = 100;
    D.iteration.n_updates = 0;


    % learning signals
    D.signals.recon_errors = [];
    D.signals.valid_errors = [];
    D.signals.lrates = [];
    D.signals.n_epochs = 0;

    % debug
    D.verbose = 0;
    D.debug.do_display = 0;
    D.debug.display_interval = 10;
    D.debug.display_fid = 1;
    D.debug.display_function = @visualize_rbm;

    % hook
    D.hook.per_epoch = {@save_intermediate, {'dae.mat'}};
    D.hook.per_update = {@print_n_updates, {}};

