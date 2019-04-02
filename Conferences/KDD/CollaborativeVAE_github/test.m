M = 300;
m_num_users = 5551;
m_num_items = 16980;

train_users = cell(m_num_users,1);
fid=fopen('data/cf-train-1-users.dat','r'); % user train file
for i=1:m_num_users
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    liked = str2num(tline);
    liked(2:end) = liked(2:end)+1;
    train_users{i} = liked;
end
fclose(fid);

test_users = cell(m_num_users,1);
fid=fopen('data/cf-test-1-users.dat','r'); % user test file
for i=1:m_num_users
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    liked = str2num(tline);
    liked(2:end) = liked(2:end)+1;
    test_users{i} = liked;
end
fclose(fid);

x = 50:50:M;
% 'cvae.mat' is the "pmf.mat" file saved by the model
%S = load('cvae.mat');
S = load('pmf.mat');
m_U = S.m_U;
m_V = S.m_V;
[recall_cvae, ~] = evaluate(train_users, test_users, m_U, m_V, M);
recall_cvae = recall_cvae(x);