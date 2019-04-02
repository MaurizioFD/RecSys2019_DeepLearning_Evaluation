function [recall] = evaluate(train_users, test_users, m_U, m_V, M)
m_num_users = size(m_U,1);
m_num_items = size(m_V,1);

batch_size = 100;
n = ceil(1.0*m_num_users/batch_size);
num_hit = zeros(m_num_users,M);
num_total = zeros(m_num_users,1);
for i=1:n
   ind = (i-1)*batch_size+1:min(i*batch_size, m_num_users);
   u_tmp = m_U(ind,:);
   score = u_tmp * m_V';
   gap = max(score(:)) - min(score(:));
   
   bs = length(ind);
   gt = zeros(bs, m_num_items);
   for j=1:bs
       idx = (i-1)*batch_size + j;
       u = train_users{idx};
       gt(j, u(2:end)) = 1;
   end
   score = score - gt*gap;
   [~,I] = sort(score, 2, 'descend');
   
   bs = length(ind);
   gt = zeros(bs, m_num_items);
   for j=1:bs
       idx = (i-1)*batch_size + j;
       u = test_users{idx};
       gt(j, u(2:end)) = 1;
   end
   re = zeros(bs, m_num_items);
   for j=1:bs
       re(j,:) = gt(j, I(j,:));
   end
   
   num_hit(ind, :) = re(:, 1:M);
   num_total(ind, :) = sum(re, 2);
end
num_hit(num_total(:,1)==0, :) = [];
num_total(num_total(:, 1)==0, :)=[];
recall = mean(cumsum(num_hit, 2)./repmat(num_total, 1, M), 1);



