% cdl - training a CDL
% Copyright (C) 2015 Hao Wang
%
% This code is based on the deepmat code by KyungHyun Cho, 
% Tapani Raiko, and Alexander Ilin
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
function [S] = cdl(S, patches, valid_patches, X_ori,valid_portion,my,perm_idx);
ratio = 1;
pid = feature('getpid'); 
mysave = my.save;
myfid = fopen(sprintf('%sCDL.log',mysave),'a');

if nargin < 3
    early_stop = 0;
    valid_patches = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end
early_stop = 0;

actual_lrate = S.learning.lrate;

n_samples = size(patches, 1);

layers = S.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = S.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = S.iteration.n_epochs;

momentum = S.learning.momentum;
weight_decay = S.learning.weight_decay;

biases_grad = cell(n_layers, 1);
W_grad = cell(n_layers, 1);
biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad{l} = zeros(size(S.biases{l}))';
    if l < n_layers
        W_grad{l} = zeros(size(S.W{l}));
    end
    biases_grad_old{l} = zeros(size(S.biases{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(S.W{l}));
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = S.do_normalize;
do_normalize_std = S.do_normalize_std;

if S.data.binary == 0
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

if S.debug.do_display == 1
    figure(S.debug.display_fid);
end

try
    use_gpu = gpuDeviceCount;
catch errgpu
    use_gpu = false;
    disp(['Could not use CUDA. Error: ' errgpu.identifier])
end

num_v = size(patches,1);
num_k = S.structure.layers(end);

if my.from~=1
    load(my.save);
end

for step=my.from:n_epochs
    if S.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
        fprintf(myfid, 'Epoch %d/%d: ', step, n_epochs)
    end

    % read m_V from final-V.dat and permute
    if step~=1
        m_V = dlmread(sprintf('%s/final-V.dat',my.save));
        m_V = m_V(perm_idx,:);
    end
    
    if use_gpu
        % push
        if my.ctrgpu && step~=1
            m_V = gpuArray(single(m_V));
        end
        for l = 1:n_layers
            if l < n_layers 
                S.W{l} = gpuArray(single(S.W{l}));
            end
            S.biases{l} = gpuArray(single(S.biases{l}));
        end

        if S.adagrad.use 
            for l = 1:n_layers
                if l < n_layers 
                    S.adagrad.W{l} = gpuArray(single(S.adagrad.W{l}));
                end
                S.adagrad.biases{l} = gpuArray(single(S.adagrad.biases{l}));
            end
        elseif S.adadelta.use
            for l = 1:n_layers
                if l < n_layers 
                    S.adadelta.gW{l} = gpuArray(single(S.adadelta.gW{l}));
                    S.adadelta.W{l} = gpuArray(single(S.adadelta.W{l}));
                end
                S.adadelta.gbiases{l} = gpuArray(single(S.adadelta.gbiases{l}));
                S.adadelta.biases{l} = gpuArray(single(S.adadelta.biases{l}));
            end
        end
    end
    

    % sdae part: update theta
    for mb=1:n_minibatches
        S.iteration.n_updates = S.iteration.n_updates + 1;

        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);
        % get minibatch of m_V
        if step~=1
            v_v = m_V((mb-1) * minibatch_sz + 1:min(mb * ...
                minibatch_sz,n_samples),:);
        end

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        if S.data.binary == 0 && S.noise.level > 0
            v0 = v0 + S.noise.level * gpuArray(randn(size(v0)));
        end

        if S.noise.drop > 0
            mask = binornd(1, 1 - S.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end

        h0e = cell(n_layers, 1);
        h0e{1} = v0;

        for l = 2:n_layers
            h0e{l} = bsxfun(@plus, h0e{l-1} * S.W{l-1}, S.biases{l}');

            if l < n_layers || S.bottleneck.binary
                h0e{l} = sigmoid(h0e{l}, S.hidden.use_tanh);
            end
            % add dropout
            if my.dropout~=0
                mask = binornd(1,1-my.dropout,size(h0e{l}));
                h0e{l} = h0e{l}.*mask;
                clear mask;
            end
        end


        % back to main code path
        h0d = cell(n_layers, 1);
        h0d{end} = h0e{end};

        for l = n_layers-1:-1:1
            h0d{l} = bsxfun(@plus, h0d{l+1} * S.W{l}', S.biases{l}');
            if l == 1 && S.data.binary
                h0d{l} = sigmoid(h0d{l});
            end
            if l > 1
                h0d{l} = sigmoid(h0d{l}, S.hidden.use_tanh);
            end
            % add dropout, mask every layer except for the last
            if my.dropout~=0 && l~=1
                mask = binornd(1,1-my.dropout,size(h0d{l}));
                h0d{l} = h0d{l}.*mask;
                clear mask;
            end
        end

        % compute reconstruction error
        hr = sdae_get_hidden(my,1, v0_clean, S);
        vr = sdae_get_visible(my, hr, S);

        if S.data.binary && S.hidden.use_tanh~=1
            rerr = -mean(sum(v0_clean .* log(max(vr, 1e-16)) + (1 - v0_clean) .* log(max(1 - vr, 1e-16)), 2));
        else
            rerr = mean(sum((v0_clean - vr).^2,2));
        end
        if use_gpu > 0
            rerr = gather(rerr);
        end
        S.signals.recon_errors = [S.signals.recon_errors rerr];

        % reset gradients
        for l = 1:n_layers
            biases_grad{l} = 0 * biases_grad{l};
            if l < n_layers
                W_grad{l} = 0 * W_grad{l};
            end
        end

        % backprop for whole net'
        deltad = cell(n_layers, 1);
        deltad{1} = h0d{1} - v0_clean;
        biases_grad{1} = mean(deltad{1}, 1);

        for l = 2:n_layers
            deltad{l} = deltad{l-1} * S.W{l-1};
            if l < n_layers || S.bottleneck.binary
                deltad{l} = deltad{l} .* dsigmoid(h0d{l}, S.hidden.use_tanh);
            end
            biases_grad{l} = mean(deltad{l}, 1);
            W_grad{l-1} = (deltad{l-1}' * h0d{l}) / (size(v0, 1));
        end

        deltae = cell(n_layers, 1);
        deltae{end} = deltad{end};

        for l = n_layers-1:-1:1
            deltae{l} = deltae{l+1} * S.W{l}';
            if l == 1 && S.data.binary
                if S.hidden.use_tanh==1
                    deltae{l} = deltae{l} .* dsigmoid(h0e{l},...
                        S.hidden.use_tanh); % added for tanh by hog
                else
                    deltae{l} = deltae{l} .* dsigmoid(h0e{l});
                end
            end
            if l > 1
                deltae{l} = deltae{l} .* dsigmoid(h0e{l}, S.hidden.use_tanh);
                biases_grad{l} = biases_grad{l} + mean(deltae{l}, 1);
            end
            W_grad{l} = W_grad{l} + (h0e{l}' * deltae{l+1}) / (size(v0, 1));
        end

        % backprop for half net'
        % first ff
        h0h = cell(n_layers, 1);
        h0h{1} = v0;

        for l = 2:n_layers
            h0h{l} = bsxfun(@plus, h0h{l-1} * S.W{l-1}, S.biases{l}');

            if l < n_layers || S.bottleneck.binary
                h0h{l} = sigmoid(h0h{l}, S.hidden.use_tanh);
            end
            % add dropout except for the bottleneck layer
            if my.dropout~=0 && l~=n_layers
                mask = binornd(1,1-my.dropout,size(h0h{l}));
                h0h{l} = h0h{l}.*mask;
                clear mask;
            end
        end

        % code digress: at Step 1, use h0e{end} as v_v
        if step==1
            v_v = h0e{end};
        end

        % back to main code path, do bp
        deltah = cell(n_layers,1);
        deltah{end} = h0h{end}-v_v;
        if S.hidden.use_tanh
            deltah{end} = deltah{end}+1;
        end
        for l = n_layers-1:-1:1
            if l~=n_layers-1
                deltah{l} = deltah{l+1}*S.W{l+1}';
            else
                deltah{l} = deltah{l+1};
            end 
            if l==1 && S.data.binary
                if S.hidden.use_tanh==1
                    deltah{l} = deltah{l}.*dsigmoid(h0h{l+1},...
                        S.hidden.use_tanh); % added for tanh by hog
                else
                    deltah{l} = deltah{l}.*dsigmoid(h0h{l+1});
                end
            end
            if l>1
                deltah{l} = deltah{l}.*dsigmoid(h0h{l+1},S.hidden.use_tanh);
            end
            biases_grad{l+1} = biases_grad{l+1}+my.lv/my.ln*mean(deltah{l},1);
            W_grad{l} = W_grad{l}+my.lv/my.ln*...
                (h0h{l}'*deltah{l})/(size(v0,1));
        end


        % learning rate
        if S.adagrad.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                if l < n_layers
                    S.adagrad.W{l} = S.adagrad.W{l} + W_grad_old{l}.^2;
                end

                S.adagrad.biases{l} = S.adagrad.biases{l} + biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                S.biases{l} = S.biases{l} - S.learning.lrate * (biases_grad_old{l}' + ...
                    weight_decay * S.biases{l}) ./ sqrt(S.adagrad.biases{l} + S.adagrad.epsilon);
                if l < n_layers
                    S.W{l} = S.W{l} - S.learning.lrate * (W_grad_old{l} + ...
                        weight_decay * S.W{l}) ./ sqrt(S.adagrad.W{l} + S.adagrad.epsilon);
                end
            end

        elseif S.adadelta.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            if S.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = S.adadelta.momentum;
            end

            for l = 1:n_layers
                if l < n_layers
                    S.adadelta.gW{l} = adamom * S.adadelta.gW{l} + (1 - adamom) * W_grad_old{l}.^2;
                end

                S.adadelta.gbiases{l} = adamom * S.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                dbias = -(biases_grad_old{l}' + ...
                    weight_decay * S.biases{l}) .* (sqrt(S.adadelta.biases{l} + S.adadelta.epsilon) ./ ...
                    sqrt(S.adadelta.gbiases{l} + S.adadelta.epsilon));
                S.biases{l} = S.biases{l} + dbias;

                S.adadelta.biases{l} = adamom * S.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                if l < n_layers
                    dW = -(W_grad_old{l} + ...
                        weight_decay * S.W{l}) .* (sqrt(S.adadelta.W{l} + S.adadelta.epsilon) ./ ...
                        sqrt(S.adadelta.gW{l} + S.adadelta.epsilon));
                    S.W{l} = S.W{l} + dW;

                    S.adadelta.W{l} = adamom * S.adadelta.W{l} + (1 - adamom) * dW.^2;

                    clear dW;
                end

            end
        else
            if S.learning.lrate_anneal > 0 && (step >= S.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if S.learning.lrate0 > 0
                    actual_lrate = S.learning.lrate / (1 + S.iteration.n_updates / S.learning.lrate0);
                else
                    actual_lrate = S.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            S.signals.lrates = [S.signals.lrates actual_lrate];

            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                S.biases{l} = S.biases{l} - actual_lrate * (biases_grad_old{l}' + weight_decay * S.biases{l});
                if l < n_layers
                    S.W{l} = S.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * S.W{l});
                end
            end
        end % end of if adagrad

        if S.verbose == 1
            fprintf(2, '.');
            fprintf(myfid, '.');
        end

        if use_gpu > 0
            clear v0 h0d h0e v0_clean vr hr deltae deltad 
        end

        if early_stop
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            v0valid = gpuArray(single(valid_patches(rndidx(1:round(n_valid * valid_portion)),:)));

            hr = sdae_get_hidden(v0valid, S);
            vr = sdae_get_visible(hr, S);

            if S.data.binary
                rerr = -mean(sum(v0valid .* log(max(vr, 1e-16)) + (1 - v0valid) .* log(max(1 - vr, 1e-16)), 2));
            else
                rerr = mean(sum((v0valid - vr).^2,2));
            end
            if use_gpu > 0
                rerr = gather(rerr);
            end

            S.signals.valid_errors = [S.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;

                if step > S.valid_min_epochs && (1.1 * valid_best_err) < valid_err
                    fprintf(2, 'Early-stop! %f, %f\n', valid_err, prev_err);
                    fprintf(myfid, 'Early-stop! %f, %f\n', valid_err, prev_err);
                    stopping = 1;
                    break;
                end

                if valid_err < valid_best_err
                    valid_best_err = valid_err;
                end
            end
        else
            if S.stop.criterion > 0
                if S.stop.criterion == 1
                    if min_recon_error > S.signals.recon_errors(end)
                        min_recon_error = S.signals.recon_errors(end);
                        min_recon_error_update_idx = S.iteration.n_updates;
                    else
                        if S.iteration.n_updates > min_recon_error_update_idx + S.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                S.signals.recon_errors(end), min_recon_error);
                            fprintf(myfid, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                S.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', S.stop.criterion);
                end
            end
        end

        if length(S.hook.per_update) > 1
            err = S.hook.per_update{1}(S, S.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if S.debug.do_display == 1 && mod(S.iteration.n_updates, S.debug.display_interval) == 0
            S.debug.display_function (S.debug.display_fid, S, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end % end of for, minibatch

    if use_gpu > 0
        % pull
        if my.ctrgpu && step~=1
            m_V = gather(m_V);
        end
        for l = 1:n_layers
            if l < n_layers
                S.W{l} = gather(S.W{l});
            end
            S.biases{l} = gather(S.biases{l});
        end

        if S.adagrad.use
            for l = 1:n_layers
                if l < n_layers
                    S.adagrad.W{l} = gather(S.adagrad.W{l});
                end
                S.adagrad.biases{l} = gather(S.adagrad.biases{l});
            end
        elseif S.adadelta.use
            for l = 1:n_layers
                if l < n_layers
                    S.adadelta.W{l} = gather(S.adadelta.W{l});
                    S.adadelta.gW{l} = gather(S.adadelta.gW{l});
                end
                S.adadelta.biases{l} = gather(S.adadelta.biases{l});
                S.adadelta.gbiases{l} = gather(S.adadelta.gbiases{l});
            end
        end
    end

    if length(S.hook.per_epoch) > 1
        err = S.hook.per_epoch{1}(S, S.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if S.verbose == 1
        fprintf(2, '\n');
        fprintf(myfid, '\n');
    end

    % ctr part
    % output theta for ctr
    m_theta = sdae_get_hidden(my,0,X_ori,S);
    if S.hidden.use_tanh
        m_theta = m_theta+1;
    end
    dlmwrite(sprintf('%s/final.gamma',my.save),m_theta,'delimiter',' ');
    % provide init V for ctr if it's the first epoch
    if step==1
        dlmwrite(sprintf('%s/final-V.dat',my.save),m_theta,'delimiter',' ');
    end

    % compose ctr cmd
    if step==n_epochs
        max_iter = my.max_iter;
    else
        max_iter = my.iter;
    end

    %fprintf('Calling external executable ./ctr ...');

%     ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ./ctr --directory %s --user ctr-data/folder%d/cf-train-1-users.dat --item ctr-data/folder%d/cf-train-1-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 1700 --random_seed 123 --theta_init %s/final.gamma >> %s/%s', ...
%         my.gsl_folder,my.save,my.folder,my.folder,max_iter,num_k,my.lv,my.lu,...
%         my.save,...
%         my.save,my.ctr_log);

    ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ./ctr --directory %s --user %s --item %s --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 1700 --random_seed 123 --theta_init %s/final.gamma >> %s/%s', ...
    my.gsl_folder,my.save,my.input_user_file,my.input_item_file,max_iter,num_k,my.lv,my.lu,...
    my.save,...
    my.save,my.ctr_log);


    %fprintf(sprintf('Executing command:\n\n %s \n\n', ctrcmd));

    if mod(step,ratio)==0 || step==n_epochs || step<5
        system(ctrcmd);
    end

    %fprintf('Calling external executable ./ctr ... done!');
        
    % compute negative log likelihood
    ctr_loss = dlmread(sprintf('%s/final-likelihood.dat',my.save));

    neg_likelihood = ...
        -ctr_loss(1,1)+...
        S.signals.recon_errors(end)*num_v*my.ln/2;

    fprintf(2, '%d: %d/%d - tre/cl/t: %4.0f/%0.4f/%f\n', pid, step, ...
        n_epochs, neg_likelihood,ctr_loss(1,1),toc);
    fprintf(myfid, '%d: %d/%d - tre/cl/t: %4.0f/%0.4f/%f\n', pid, ...
        step, n_epochs, neg_likelihood,ctr_loss(1,1),toc);

    % save tmp result according to save_lag
    if mod(step,my.save_lag)==0
        system(sprintf('cp %s/final-V.dat %s/%d-V.dat',my.save,...
            my.save,step));
        system(sprintf('cp %s/final-U.dat %s/%d-U.dat',my.save,...
            my.save,step));
    end
end % end of for, n_epoch

if use_gpu > 0
    % pull
    for l = 1:n_layers
        if l < n_layers
            S.W{l} = gather(S.W{l});
        end
        S.biases{l} = gather(S.biases{l});
    end

    if S.adagrad.use
        for l = 1:n_layers
            if l < n_layers
                S.adagrad.W{l} = gather(S.adagrad.W{l});
            end
            S.adagrad.biases{l} = gather(S.adagrad.biases{l});
        end
    elseif S.adadelta.use
        for l = 1:n_layers
            if l < n_layers
                S.adadelta.W{l} = gather(S.adadelta.W{l});
                S.adadelta.gW{l} = gather(S.adadelta.gW{l});
            end
            S.adadelta.biases{l} = gather(S.adadelta.biases{l});
            S.adadelta.gbiases{l} = gather(S.adadelta.gbiases{l});
        end
    end
end
fclose(myfid);

