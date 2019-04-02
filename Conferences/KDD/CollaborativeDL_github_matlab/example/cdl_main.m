function cld_main
para_gpuDevice='no';
para_layers=[8000 200 50];
para_blayers=[1 1 1];
para_lv=10;
para_lu=1;
para_ln=1e3;
para_pretrain=0;
para_save=999;
para_folder=45;
para_dropout=0.1;
para_from=1;
%para_sdae_n_epoch=1000;
para_sdae_n_epoch=7;
para_save_lag=501;

temp_folder = '../../../../result_experiments/__Temp_CollaborativeDL_Matlab_RecommenderWrapper/';
gsl_folder = '/usr/lib/x86_64-linux-gnu/';

% input_user_file = 'ctr-data/folder45/cf-train-1-users.dat';
% input_item_file = 'ctr-data/folder45/cf-train-1-items.dat';
% content_file = 'mult_nor.mat';

input_user_file = strcat(temp_folder,'cf-train-users.dat');
input_item_file = strcat(temp_folder,'cf-train-items.dat');
content_file = strcat(temp_folder,'mult_nor.mat');


fprintf('The pid is: %d\n',feature('getpid'));
cdl_worker(para_gpuDevice,para_layers,para_blayers,...
    para_lv,para_lu,para_ln,para_pretrain,para_save,...
    para_folder,para_dropout,para_from,para_sdae_n_epoch,...
    para_save_lag,temp_folder, gsl_folder, input_user_file, input_item_file,content_file);

%exit;
