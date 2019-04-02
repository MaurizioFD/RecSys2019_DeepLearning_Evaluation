% dae - training a single-layer DAE
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
function [D] = dae(my, D, patches, valid_patches, valid_portion);
D.mystopping = -1;
pid = feature('getpid');

my.fid = fopen(sprintf('%s.log',my.save),'a');
if nargin < 4
    early_stop = 0;
    valid_patches = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end

actual_lrate = D.learning.lrate;

n_samples = size(patches, 1);
if D.structure.n_visible ~= size(patches, 2)
    error('Data is not properly aligned');
end

vbias_grad_old = zeros(size(D.vbias'));
hbias_grad_old = zeros(size(D.hbias'));
W_grad_old = zeros(size(D.W));

minibatch_sz = D.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = D.iteration.n_epochs;

momentum = D.learning.momentum;
weight_decay = D.learning.weight_decay;

n_visible = D.structure.n_visible;
n_hidden = D.structure.n_hidden;

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;
D.mystopping = 0;

do_normalize = D.do_normalize;
do_normalize_std = D.do_normalize_std;

if D.data.binary == 0
    if do_normalize == 1
        % make it zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
    end

    if do_normalize_std ==1
        % make it unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
    end
end

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if D.debug.do_display == 1
    figure(D.debug.display_fid);
end

try
    use_gpu = gpuDeviceCount;
catch errgpu
    use_gpu = false;
    disp(['Could not use CUDA. Error: ' errgpu.identifier])
end

for step=1:n_epochs
    if D.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
        fprintf(my.fid, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        D.W = gpuArray(single(D.W));
        D.vbias = gpuArray(single(D.vbias));
        D.hbias = gpuArray(single(D.hbias));

        if D.adagrad.use
            D.adagrad.W = gpuArray(single(D.adagrad.W));
            D.adagrad.vbias = gpuArray(single(D.adagrad.vbias));
            D.adagrad.hbias = gpuArray(single(D.adagrad.hbias));
        elseif D.adadelta.use
            D.adadelta.gW = gpuArray(single(D.adadelta.gW));
            D.adadelta.gvbias = gpuArray(single(D.adadelta.gvbias));
            D.adadelta.ghbias = gpuArray(single(D.adadelta.ghbias));

            D.adadelta.W = gpuArray(single(D.adadelta.W));
            D.adadelta.vbias = gpuArray(single(D.adadelta.vbias));
            D.adadelta.hbias = gpuArray(single(D.adadelta.hbias));
        end
    end

    for mb=1:n_minibatches
        D.iteration.n_updates = D.iteration.n_updates + 1;

        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        if D.data.binary == 0 && D.noise.level > 0
            v0 = v0 + D.noise.level * gpuArray(randn(size(v0)));
        end

        if D.noise.drop > 0
            mask = binornd(1, 1 - D.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end

        h0 = bsxfun(@plus, v0 * D.W, D.hbias');
        if D.hidden.binary
            h0 = sigmoid(h0, D.hidden.use_tanh);
        end
        % dropout
        if my.dropout~=0
            mask = binornd(1,1-my.dropout,size(h0));
            h0 = h0.*mask;
            clear mask;
        end

        % compute reconstruction error
        hr = bsxfun(@plus, v0_clean * D.W, D.hbias');
        % recover from dropout
        if my.dropout~=0
            hr = hr.*(1-my.dropout);
        end
        if D.hidden.binary
            hr = sigmoid(hr, D.hidden.use_tanh);
        end

        vr = bsxfun(@plus,hr * D.W',D.vbias');
        if D.data.binary
            vr = sigmoid(vr, D.visible.use_tanh);
        end

        if D.data.binary && ~D.visible.use_tanh
            rerr = -mean(sum(v0_clean .* log(max(vr, 1e-16)) + (1 - v0_clean) .* log(max(1 - vr, 1e-16)), 2));
        else
            rerr = mean(sum((v0_clean - vr).^2,2));
        end
        if use_gpu > 0
            rerr = gather(rerr);
        end
        D.signals.recon_errors = [D.signals.recon_errors rerr];

        % get gradient
        vr = bsxfun(@plus,h0 * D.W',D.vbias');
        if D.data.binary
            vr = sigmoid(vr, D.visible.use_tanh);
        end

        deltao = vr - v0_clean;
        if D.data.binary && D.visible.use_tanh
            deltao = deltao .* dsigmoid(vr, D.visible.use_tanh);
        end
        vbias_grad = mean(deltao, 1);

        clear hr vr;

        deltah = deltao * D.W;
        if D.hidden.binary
            deltah = deltah .* dsigmoid(h0, D.hidden.use_tanh);
        end
        hbias_grad = mean(deltah, 1);

        W_grad = ((deltao' * h0) + (v0' * deltah)) / size(v0,1);

        clear deltao deltah;

        if D.sparsity.cost > 0 && D.hidden.use_tanh == 0
            diff_sp = (h0 - D.sparsity.target);
            hbias_grad = hbias_grad + D.sparsity.cost * mean(diff_sp, 1);
            %W_grad = W_grad + (D.sparsity.cost/size(v0,1)) * (v0_clean' * diff_sp);
            W_grad = W_grad + (D.sparsity.cost/size(v0,1)) * (v0' * diff_sp);
            clear diff_sp;
        end

        if D.cae.cost > 0 && D.hidden.use_tanh == 0
            W_cae1 = bsxfun(@times, D.W, mean(h0 .* (1 - h0).^2, 1));
            W_cae2 = D.W.^2 .* (v0' * (...
                (1 - 2 * h0) .* h0 .* (1 - h0).^2 ...
            ) / size(v0, 1));

            W_cae = W_cae1 + W_cae2;
            W_grad = W_grad + D.cae.cost * W_cae;

            clear W_cae1 W_cae2 W_cae;

            hbias_cae = sum(bsxfun(@times, D.W, mean(h0 .* (1 - h0).^2 .* (1 - 2 * h0),1)), 1);
            hbias_grad = hbias_grad + D.cae.cost * hbias_cae;

            clear hbias_cae;
        end

        if D.rica.cost > 0
            W_rica = v0_clean' * tanh(hr);
            W_grad = W_grad + D.rica.cost * W_rica;

            clear W_rica;

%            hbias_rica = mean(tanh(hr), 1);
%            hbias_grad = hbias_grad + D.rica.cost * hbias_rica;

%            clear W_rica;
        end

        if D.adagrad.use
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;

            D.adagrad.W = D.adagrad.W + W_grad_old.^2;
            D.adagrad.vbias = D.adagrad.vbias + vbias_grad_old.^2';
            D.adagrad.hbias = D.adagrad.hbias + hbias_grad_old.^2';


            if D.rica.cost <= 0
                D.vbias = D.vbias - D.learning.lrate * (vbias_grad_old' + weight_decay * D.vbias) ./ sqrt(D.adagrad.vbias + D.adagrad.epsilon);
                D.hbias = D.hbias - D.learning.lrate * (hbias_grad_old' + weight_decay * D.hbias) ./ sqrt(D.adagrad.hbias + D.adagrad.epsilon);
            end

            D.W = D.W - D.learning.lrate * (W_grad_old + weight_decay * D.W) ./ sqrt(D.adagrad.W + D.adagrad.epsilon);
        elseif D.adadelta.use
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;

            if D.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = D.adadelta.momentum;
            end

            D.adadelta.gW = adamom * D.adadelta.gW + (1 - adamom) * W_grad_old.^2;
            D.adadelta.gvbias = adamom * D.adadelta.gvbias + (1 - adamom) * vbias_grad_old.^2';
            D.adadelta.ghbias = adamom * D.adadelta.ghbias + (1 - adamom) * hbias_grad_old.^2';

            if D.rica.cost <= 0
                dvbias = -(vbias_grad_old' + ...
                    weight_decay * D.vbias) .* (sqrt(D.adadelta.vbias + D.adadelta.epsilon) ./ sqrt(D.adadelta.gvbias + D.adadelta.epsilon));
                dhbias = -(hbias_grad_old' + ...
                    weight_decay * D.hbias) .* (sqrt(D.adadelta.hbias + D.adadelta.epsilon) ./ sqrt(D.adadelta.ghbias + D.adadelta.epsilon));

                D.vbias = D.vbias + dvbias;
                D.hbias = D.hbias + dhbias;
            end

            dW = -(W_grad_old + weight_decay * D.W) .* ...
                (sqrt(D.adadelta.W + D.adadelta.epsilon) ./ sqrt(D.adadelta.gW + D.adadelta.epsilon));
            D.W = D.W + dW;

            D.adadelta.W = adamom * D.adadelta.W + (1 - adamom) * dW.^2;
            clear dW;

            if D.rica.cost <= 0
                D.adadelta.vbias = adamom * D.adadelta.vbias + (1 - adamom) * dvbias.^2;
                D.adadelta.hbias = adamom * D.adadelta.hbias + (1 - adamom) * dhbias.^2;
                clear dvbias dhbias;
            end
        else
            if D.learning.lrate_anneal > 0 && (step >= D.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if D.learning.lrate0 > 0
                    actual_lrate = D.learning.lrate / (1 + D.iteration.n_updates / D.learning.lrate0);
                else
                    actual_lrate = D.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            D.signals.lrates = [D.signals.lrates actual_lrate];

            % update
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;

            if D.rica.cost <= 0
                D.vbias = D.vbias - actual_lrate * (vbias_grad_old' + weight_decay * D.vbias);
                D.hbias = D.hbias - actual_lrate * (hbias_grad_old' + weight_decay * D.hbias);
            end

            D.W = D.W - actual_lrate * (W_grad_old + weight_decay * D.W);
        end

        if D.verbose == 1
            fprintf(2, '.');
            fprintf(my.fid, '.');
        end

        if use_gpu > 0
            clear v0 h0 v0_clean vr hr deltao deltah 
        end

        if early_stop
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            v0valid = valid_patches(rndidx(1:round(n_valid * valid_portion)),:);
            if use_gpu > 0
                v0valid = gpuArray(single(v0valid));
            end

            hr = bsxfun(@plus, v0valid * D.W, D.hbias');
            % recover from dropout
            if my.dropout~=0
                hr = hr.*(1-my.dropout);
            end
            if D.hidden.binary
                hr = sigmoid(hr, D.hidden.use_tanh);
            end

            vr = bsxfun(@plus,hr * D.W',D.vbias');
            if D.data.binary
                vr = sigmoid(vr, D.visible.use_tanh);
            end

            if D.data.binary && ~D.visible.use_tanh
                rerr = -mean(sum(v0valid .* log(max(vr, 1e-16)) + (1 - v0valid) .* log(max(1 - vr, 1e-16)), 2));
            else
                rerr = mean(sum((v0valid - vr).^2,2));
            end
            if use_gpu > 0
                rerr = gather(rerr);
            end

            D.signals.valid_errors = [D.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;

                if step > D.valid_min_epochs && (my.early_stop_thre * valid_best_err) < valid_err
                    fprintf(2, 'Early-stop! %f, %f\n', valid_err, prev_err);
                    fprintf(my.fid, 'Early-stop! %f, %f\n', valid_err, prev_err);

                    stopping = 1;
                    break;
                end

                if valid_err < valid_best_err
                    valid_best_err = valid_err;
                end
            end
        else
            if D.stop.criterion > 0
                if D.stop.criterion == 1
                    if min_recon_error > D.signals.recon_errors(end)
                        min_recon_error = D.signals.recon_errors(end);
                        min_recon_error_update_idx = D.iteration.n_updates;
                    else
                        if D.iteration.n_updates > min_recon_error_update_idx + D.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                D.signals.recon_errors(end), min_recon_error);
                            fprintf(my.fid, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                D.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', D.stop.criterion);
                end
            end
        end

        if length(D.hook.per_update) > 1
            err = D.hook.per_update{1}(D, D.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if D.debug.do_display == 1 && mod(D.iteration.n_updates, D.debug.display_interval) == 0
            D.debug.display_function (D.debug.display_fid, D, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end

    if use_gpu > 0
        % pull
        D.W = gather(D.W);
        D.vbias = gather(D.vbias);
        D.hbias = gather(D.hbias);

        if D.adagrad.use
            D.adagrad.W = gather(D.adagrad.W);
            D.adagrad.vbias = gather(D.adagrad.vbias);
            D.adagrad.hbias = gather(D.adagrad.hbias);
        elseif D.adadelta.use
            D.adadelta.W = gather(D.adadelta.W);
            D.adadelta.vbias = gather(D.adadelta.vbias);
            D.adadelta.hbias = gather(D.adadelta.hbias);

            D.adadelta.gW = gather(D.adadelta.gW);
            D.adadelta.gvbias = gather(D.adadelta.gvbias);
            D.adadelta.ghbias = gather(D.adadelta.ghbias);
        end
    end

    if length(D.hook.per_epoch) > 1
        err = D.hook.per_epoch{1}(D, D.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        D.mystopping = step;
        break;
    end
    
    if D.verbose == 1
        fprintf(2, '\n');
        fprintf(my.fid, '\n');
    end
        
    %fprintf(2, 'Epoch %d/%d - recon_error: %f norms: %f/%f/%f/%0.4f\n', step, n_epochs, ...
    %    D.signals.recon_errors(end), ...
    %    D.W(:)' * D.W(:) / length(D.W(:)), ...
    %    D.vbias' * D.vbias / length(D.vbias), ...
    %    D.hbias' * D.hbias / length(D.hbias), ...
    %    valid_err);
    fprintf(2, '%d: %d/%d - rec_e: %f res: %f/%f/%0.4f\n', pid, step, n_epochs, ...
        D.signals.recon_errors(end), ...
        toc, ...
        D.hbias' * D.hbias / length(D.hbias), ...
        valid_err);
    fprintf(my.fid, '%d: %d/%d - rec_e: %f res: %f/%f/%0.4f\n', pid, step, n_epochs, ...
        D.signals.recon_errors(end), ...
        toc, ...
        D.hbias' * D.hbias / length(D.hbias), ...
        valid_err);
end

if use_gpu > 0
    % pull
    D.W = gather(D.W);
    D.vbias = gather(D.vbias);
    D.hbias = gather(D.hbias);

    if D.adagrad.use
        D.adagrad.W = gather(D.adagrad.W);
        D.adagrad.vbias = gather(D.adagrad.vbias);
        D.adagrad.hbias = gather(D.adagrad.hbias);
    elseif D.adadelta.use
        D.adadelta.W = gather(D.adadelta.W);
        D.adadelta.vbias = gather(D.adadelta.vbias);
        D.adadelta.hbias = gather(D.adadelta.hbias);

        D.adadelta.gW = gather(D.adadelta.gW);
        D.adadelta.gvbias = gather(D.adadelta.gvbias);
        D.adadelta.ghbias = gather(D.adadelta.ghbias);
    end
end
fclose(my.fid);


