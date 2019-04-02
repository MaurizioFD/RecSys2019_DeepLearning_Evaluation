% default_dbm - 
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
function [S] = default_sdae (layers)
    % data type
    S.data.binary = 1;
    %S.data.binary = 0; % for GDBM

    % bottleneck layer
    S.bottleneck.binary = 1;
    %S.bottleneck.binary = 0;

    % nonlinearity: the name of the variable will change in the later revision
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    S.hidden.use_tanh = 0;
    S.visible.use_tanh = 0; % added by hog

    % learning parameters
    S.learning.lrate = 1e-3;
    S.learning.lrate0 = 5000;
    S.learning.momentum = 0;
    S.learning.weight_decay = 0;
    S.learning.minibatch_sz = 100;
    S.learning.lrate_anneal = 0.9;

    S.valid_min_epochs = 10;

    S.do_normalize = 1;
    S.do_normalize_std = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    S.stop.criterion = 0;
    % criterion == 1
    S.stop.recon_error.tolerate_count = 1000;

    % denoising
    S.noise.drop = 0.1;
    S.noise.level = 0.1;

    % structure
    n_layers = length(layers);
    S.structure.layers = layers;

    % initializations
    S.W = cell(n_layers, 1);
    S.biases = cell(n_layers, 1);
    for l = 1:n_layers
        S.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            %S.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
            S.W{l} = 2 * sqrt(6)/sqrt(layers(l)+layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
        end
    end

    % adagrad
    S.adagrad.use = 0;
    S.adagrad.epsilon = 1e-8;
    S.adagrad.W = cell(n_layers, 1);
    S.adagrad.biases = cell(n_layers, 1);
    for l = 1:n_layers
        S.adagrad.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            S.adagrad.W{l} = zeros(layers(l), layers(l+1));
        end
    end

    S.adadelta.use = 0;
    S.adadelta.momentum = 0.99;
    S.adadelta.epsilon = 1e-6;
    S.adadelta.gW = cell(n_layers, 1);
    S.adadelta.gbiases = cell(n_layers, 1);
    S.adadelta.W = cell(n_layers, 1);
    S.adadelta.biases = cell(n_layers, 1);
    for l = 1:n_layers
        S.adadelta.gbiases{l} = zeros(layers(l), 1);
        S.adadelta.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            S.adadelta.gW{l} = zeros(layers(l), layers(l+1));
            S.adadelta.W{l} = zeros(layers(l), layers(l+1));
        end
    end

    % iteration
    S.iteration.n_epochs = 100;
    S.iteration.n_updates = 0;

    % learning signals
    S.signals.recon_errors = [];
    S.signals.valid_errors = [];
    S.signals.lrates = [];
    S.signals.n_epochs = 0;

    % debug
    S.verbose = 0;
    S.debug.do_display = 0;
    S.debug.display_interval = 10;
    S.debug.display_fid = 1;
    S.debug.display_function = @visualize_dae;

    % hook
    S.hook.per_epoch = {@save_intermediate, {'sdae.mat'}};
    S.hook.per_update = {@print_n_updates, {}};

