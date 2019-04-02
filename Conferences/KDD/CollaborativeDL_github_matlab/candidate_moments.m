costs = [];
cand_moments = [];
cand_moment = base_moment;
for s=1:(max_iter_up + 1)
    cand_moments = [cand_moments cand_moment];
    cand_moment = cand_moment * adaptive_moment_exp_up;
end
cand_moment = base_moment * adaptive_moment_exp_down;
for s=1:(max_iter_down)
    cand_moments = [cand_moments cand_moment];
    cand_moment = cand_moment * adaptive_moment_exp_down;
end


