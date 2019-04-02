function mult_sdae_worker(gd,pls,bls,lv,lu,ln,prtrn,sv,fd,dpt,fr,n_epo_sdae,n_epo_dae,sl,temp_folder,...
    gsl_folder, input_user_file, input_item_file, content_file, minibatch_size)
fprintf('The pid is %d\n',feature('getpid'));
use_tanh = 0; % remember to change back to 0 with the hidden+1 in sdae.m
my.from = fr;
my.lv = lv;
my.lu = lu;
my.ln = ln;
my.noise.drop = 0.3;
my.sparsity.cost = 0.1;
if prtrn==0
    my.pretrain = 0;
else
    my.pretrain = temp_folder; % pretrain
end
rand('seed',11112);
%my.save = sprintf('cdl%d',sv);
my.save = temp_folder;
my.input_user_file = input_user_file;
my.input_item_file = input_item_file;
my.gsl_folder = gsl_folder;
my.folder = fd;
my.weight_decay = 1e-4; % default is 1e-4
my.dropout = dpt;
my.iter = 1;
do_pretrain = 1;

if gd~='no'
    gpuDevice;
end

%gpuDevice(gd);
%gpuDevice;

my.save_lag = sl;
my.sdae.n_epoch = n_epo_sdae;
%my.dae.n_epoch = 500;
my.dae.n_epoch = n_epo_dae;
my.max_iter = 100;
my.minibatch_size = minibatch_size;
% if my.folder>44 && my.folder<50
%     my.minibatch_size = 128;
% elseif my.folder>39 && my.folder<45
%     my.minibatch_size = 256;
% elseif my.folder>84 && my.folder<90
%     my.minibatch_size = 128;
% elseif my.folder>64 && my.folder<70
%     my.minibatch_size = 128;
% else
%     my.minibatch_size = 128;
% end


my.a = 1;
my.b = 0.01;
my.ctrgpu = 1;
if my.folder>44 && my.folder<50
    my.graph = 'S-rand000a.dat';
elseif my.folder>39 && my.folder<45
    my.graph = 'S-rand000b.dat';
elseif my.folder>84 && my.folder<90
    my.graph = 'S-rand000m.dat';
elseif my.folder>64 && my.folder<70
    my.graph = 'S-rand000n.dat';
else
    my.graph = 'S-rand000n.dat';
end
my.ctr_log = 'ctr.log';
%my.ctr_log = sprintf('%s/ctr.log',my.save);
my.adagrad.use = 0;
my.early_stop_thre = 2;
% add the path of RBM code
addpath('..');
mkdir(my.save);

% added by hog
mystopping = [-2];

% load content

load(content_file);

% if my.folder>44 && my.folder<50
%     load 'mult_nor.mat';
%     %load 'mult_nor3.mat'; % only titles, 14496 words
% elseif my.folder>39 && my.folder<45
%     load 'mult_nor2.mat';
% elseif my.folder>84 && my.folder<90
%     load 'mult_movie_large.mat';
% elseif my.folder>64 && my.folder<70
%     load 'mult_netflix_large.mat';
% else
%     load 'mult_netflix_large.mat';
% end

if use_tanh==1
    X = (X-0.5)*2;
    my.adadelta_use = 0;
    my.learning_rate = 1e-6;
    my.d.adadelta_use = 0;
    my.d.learning_rate = 1e-5;
else
    my.adadelta_use = 1;
    my.learning_rate = 1e-1;
    my.d.adadelta_use = 1;
    my.d.learning_rate = 1e-1;
end

% shuffle the training data
perm_idx = randperm(size(X,1));

n_all = size(X, 1);
n_pre_train = ceil(n_all * 3 / 4);
n_pre_valid = floor(n_all /4);
n_train = n_all;
n_valid = 0;

X_ori = X;
X_valid = X(perm_idx(n_train+1:end), :);
X = X(perm_idx(1:n_train), :);
X_pre_valid = X(perm_idx(n_pre_train+1:end), :);
X_pre = X(perm_idx(1:n_pre_train), :);

layers = pls;
blayers = bls;
n_layers = length(layers);


if my.pretrain
    load(my.pretrain);
    fprintf('using pretrain result: %s\n',my.pretrain);
end

if do_pretrain && ~isstr(my.pretrain)
    fprintf('doing pretrain from scratch\n');
    my.fid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(my.fid, 'doing pretrain from scratch\n');
    fclose(my.fid);

    Ds = cell(n_layers - 1, 1);
    H = X_pre;
    H_valid = X_pre_valid;

    for l = 1:n_layers-1
        % construct DAE and use default configurations
        D = default_dae (layers(l), layers(l+1));

        D.data.binary = blayers(l);
        D.hidden.binary = blayers(l+1);

        if use_tanh 
            if l > 1
                D.visible.use_tanh = 1;
            else
                D.visible.use_tanh = 1; % added by hog
            end
            D.hidden.use_tanh = 1;
        else
            if D.data.binary
                mH = mean(H, 1)';
                D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            else
                D.vbias = mean(H, 1)';
            end
        end

        D.learning.lrate = my.d.learning_rate; % default: 1e-1
        D.learning.lrate0 = 5000;
        D.learning.weight_decay = 0.0001;
        D.learning.minibatch_sz = my.minibatch_size;

        D.valid_min_epochs = 10;

        %D.noise.drop = 0.2; deleted by hog
        D.noise.drop = my.noise.drop;   % added by hog
        D.sparsity.cost = my.sparsity.cost;        % added by hog
        D.noise.level = 0;

        %D.adagrad.use = 1;
        %D.adagrad.epsilon = 1e-8;
        D.adagrad.use = my.adagrad.use;
        D.adadelta.use = my.d.adadelta_use;
        D.adadelta.epsilon = 1e-8;
        D.adadelta.momentum = 0.99;

        D.iteration.n_epochs = my.dae.n_epoch;

        % save the intermediate data after every epoch
        D.hook.per_epoch = {@save_intermediate, {sprintf('dae_mult_%d.mat', l)}};

        % print learining process
        D.verbose = 0;
        % display the progress
        D.debug.do_display = 0;

        % train RBM
        my.fid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(my.fid, 'Training DAE (%d)\n', l);
        fclose(my.fid);
        fprintf(1, 'Training DAE (%d)\n', l);

        tic;
        D = dae (my, D, H,H_valid,0.1);
        mystopping = [mystopping D.mystopping];

        my.fid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(1, 'Training is done after %f seconds\n', toc);
        fprintf(my.fid, 'Training is done after %f seconds\n', toc);
        fclose(my.fid);

        H = dae_get_hidden(H, D);
        H_valid = dae_get_hidden(H_valid, D);

        Ds{l} = D;
    end % end of for nlayers
end

S = default_sdae (layers);
S.mystopping = mystopping;

S.data.binary = blayers(1);
S.bottleneck.binary = blayers(end);
S.hidden.use_tanh = use_tanh;

S.hook.per_epoch = {@save_intermediate, {my.save}};

S.learning.lrate = 1e-6; % default: 1e-1
S.learning.lrate0 = 5000;
%S.learning.momentum = 0.9;
S.learning.weight_decay = my.weight_decay;
S.learning.minibatch_sz = my.minibatch_size;

%S.noise.drop = 0.2;
%S.noise.level = 0;
S.noise.drop = my.noise.drop;   % added by hog
S.sparsity.cost = 0.1;          % added by hog
S.adadelta.use = my.adadelta_use;
S.adadelta.epsilon = 1e-8; % default: 1e-8
S.adadelta.momentum = 0.99; % default: 0.99

%S.adagrad.use = 1;
%S.adagrad.epsilon = 1e-8;
S.valid_min_epochs = 10;

S.iteration.n_epochs = my.sdae.n_epoch;

if do_pretrain && my.from==1
    for l = 1:n_layers-1
        S.biases{l+1} = Ds{l}.hbias;
        S.W{l} = Ds{l}.W;
    end
elseif my.from==1
    fprintf('initialize sdae using mean\n');
    my.fid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(my.fid, 'initialize sdae using mean\n');
    fclose(my.fid);
    if S.data.binary
        mH = mean(X, 1)';
        S.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);
    else
        S.biases{1} = mean(X, 1)';
    end
end

my.fid = fopen(sprintf('%s.log',my.save),'a');
fprintf(my.fid, 'Training CDL\n');
fclose(my.fid);

fprintf(1, 'Training CDL\n');
tic;
S = cdl (S, X, X_valid, X_ori, 0.1,my,perm_idx);

my.fid = fopen(sprintf('%s.log',my.save),'a');
fprintf(my.fid, 'Training is done after %f seconds\n', toc);
fclose(my.fid);

fprintf(1, 'Training is done after %f seconds\n', toc);

X = X_ori;
H = sdae_get_hidden (my,0,X, S);
H_valid = sdae_get_hidden(my,0,X_valid,S);
save(sprintf('%ssdae_hidden',my.save),'H','S','Ds');

fprintf(1, 'Saved hidden data. Terminating\n');






