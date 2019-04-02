function cld_main(temp_folder, gsl_folder, input_user_file, input_item_file, content_file, ...
                    para_lv, para_lu, para_ln, para_sdae_n_epoch, para_dae_n_epoch, ...
                    para_pretrain, minibatch_size, n_features)
para_gpuDevice='no';

n_features = double(n_features);
para_layers=[n_features 200 50];
para_blayers=[1 1 1];
%para_lv=10;
%para_lu=1;
%para_ln=1e3;
para_lv=double(para_lv);
para_lu=double(para_lu);
para_ln=double(para_ln);

para_pretrain=0;
para_save=999;
para_folder=45;
para_dropout=0.1;
para_from=1;
%para_sdae_n_epoch=1000;
%para_sdae_n_epoch=7;
para_save_lag=501;

fprintf('The pid is: %d\n',feature('getpid'));
cdl_worker(para_gpuDevice,para_layers,para_blayers,...
    para_lv,para_lu,para_ln,para_pretrain,para_save,...
    para_folder,para_dropout,para_from,para_sdae_n_epoch,para_dae_n_epoch,...
    para_save_lag,temp_folder, gsl_folder, input_user_file, input_item_file,...
    content_file,minibatch_size);

fprintf('cdl_worker call complete');
