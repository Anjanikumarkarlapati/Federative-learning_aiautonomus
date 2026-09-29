function results = fl_prototype(varargin)
%FL_PROTOTYPE Federated learning prototype for collaborative autonomous-vehicle training.
%
%   WHAT THIS DEMONSTRATES
%   A fleet of N vehicles jointly trains ONE shared model without any vehicle
%   sharing raw sensor data. Each round:
%     1. SELECT    the Road-Side Unit (RSU) picks a fraction C of the fleet.
%     2. TRAIN     each chosen vehicle runs a few epochs of SGD on private data.
%     3. UPLOAD    the vehicle sends only dW_k = w_k - w (optionally clipped and
%                  noised for privacy, and top-k sparsified to save bandwidth).
%     4. AGGREGATE w <- w + sum_k a_k dW_k,  a_k = q_k n_k / sum_j q_j n_j
%                  n_k = #samples, q_k = quality score of vehicle k.
%
%   The model is a softmax classifier on synthetic sensor features, a stand-in
%   for the CNN in the full Python framework (fedav/). Mirrors
%   python_prototype/fl_prototype.py. Base MATLAB only, no toolboxes.
%
%   WHY THE DATA IS HARD (non-IID)
%     label skew      p_k ~ Dirichlet(alpha)         different class mixes
%     quantity skew   n_k ~ LogNormal                very different data amounts
%     feature shift   per-vehicle offset             region / weather / time of day
%     sensor noise    per-vehicle noise level        degraded sensors
%     connectivity    per-vehicle uptime in [.2, 1]  unstable V2I links drop uploads
%
%   ARMS
%     centralized     upper bound, pooled data (violates privacy; reference only)
%     fedavg          random selection, size-weighted average
%     +selection      connectivity-aware selection only
%     +quality        quality-aware aggregation only
%     proposed        both of the above
%     proposed+topk   top-k sparsified uploads with error feedback
%     proposed+dp     per-update L2 clipping + Gaussian noise
%
%   USAGE
%     fl_prototype                              % defaults
%     fl_prototype('clients',40,'rounds',50,'repeats',5)
%     fl_prototype('verbose',false)             % summary only
%     fl_prototype('plot',false)                % no figure
%   Options: clients(20) rounds(30) classes(6) dim(16) alpha(0.4) frac(0.3)
%            seed(1) repeats(3) verbose(true) plot(true)
%
%   Output: struct with .summary (table) and .history (per-round, first seed).

p = inputParser;
addParameter(p, 'clients', 20);  addParameter(p, 'rounds', 30);
addParameter(p, 'classes', 6);   addParameter(p, 'dim', 16);
addParameter(p, 'alpha', 0.4);   addParameter(p, 'frac', 0.3);
addParameter(p, 'seed', 1);      addParameter(p, 'repeats', 3);
addParameter(p, 'verbose', true); addParameter(p, 'plot', true);
parse(p, varargin{:});
o = p.Results;

fprintf('FL-AV prototype: N=%d vehicles, R=%d rounds, C=%.2f, Dirichlet alpha=%.2f, repeats=%d\n', ...
    o.clients, o.rounds, o.frac, o.alpha, o.repeats);

armNames = {'centralized', 'fedavg', '+selection', '+quality', 'proposed', 'proposed+topk', 'proposed+dp'};
nArms = numel(armNames);
% metric columns: acc, loss, worst-vehicle acc, rounds-to-90%, uploaded KB, dropped
M = zeros(o.repeats, nArms, 6);
firstHist = cell(1, nArms);

for k = 1:o.repeats
    seed = o.seed + k - 1;
    rng(seed);
    [clients, Xt, yt] = make_fleet(o.clients, o.classes, o.dim, o.alpha, 150);
    show = o.verbose && k == 1;
    if show, describe_fleet(clients, o.classes); fprintf('\nTRAINING (per-round log for the proposed method)\n'); end

    for a = 1:nArms
        if a == 1
            [W, h] = run_central(clients, Xt, yt, o.dim, o.classes, o.rounds, seed);
        else
            [W, h] = run_fl(clients, Xt, yt, o.dim, o.classes, o.rounds, o.frac, seed, ...
                arm_options(armNames{a}), show && strcmp(armNames{a}, 'proposed'), armNames{a});
        end
        pv = per_vehicle_accuracy(W, clients);
        r90 = find(h(:,3) >= 0.9 * h(end,3), 1);
        M(k, a, :) = [h(end,3), h(end,2), min(pv), r90, sum(h(:,6)), sum(h(:,5))];
        if k == 1, firstHist{a} = h; end
    end
end

Mm = squeeze(mean(M, 1)); Ms = squeeze(std(M, 0, 1));
if o.repeats == 1, Ms = zeros(size(Mm)); end
fprintf('\nRESULTS (mean +/- std over %d seeds; global clean test set)\n', o.repeats);
fprintf('  %-15s %16s %14s %14s %12s %8s\n', 'arm', 'accuracy', 'worst vehicle', 'rounds to 90%', 'uploaded KB', 'dropped');
for a = 1:nArms
    fprintf('  %-15s %8.3f +/-%.3f %14.3f %14.1f %12.2f %8.1f\n', armNames{a}, ...
        Mm(a,1), Ms(a,1), Mm(a,3), Mm(a,4), Mm(a,5), Mm(a,6));
end
fprintf('\nHow to read this:\n');
fprintf('  * centralized is the ceiling; it needs all raw data on one server.\n');
fprintf('  * fedavg -> +selection / +quality isolates what each ingredient adds; differences\n');
fprintf('    can be small next to the std across seeds, so use ''repeats'',10+ before concluding.\n');
fprintf('  * proposed+topk: the toy model has few weights, so index overhead limits the saving;\n');
fprintf('    a model with millions of weights amortises it.\n');
fprintf('  * proposed+dp shows the accuracy price of the privacy noise.\n');
fprintf('  * ''worst vehicle'' = accuracy of the global model on the hardest vehicle''s data.\n');

% ---- files ---------------------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'rounds.csv'), 'w');
fprintf(fid, 'arm,round,test_loss,test_acc,participants,dropped,uploaded_kb\n');
for a = 1:nArms
    h = firstHist{a};
    for r = 1:size(h, 1)
        fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%.6f\n', armNames{a}, h(r,1), h(r,2), h(r,3), h(r,4), h(r,5), h(r,6));
    end
end
fclose(fid);
fid = fopen(fullfile(outDir, 'summary.csv'), 'w');
fprintf(fid, 'arm,acc_mean,acc_std,worst_vehicle_acc,rounds_to_90pct,uploaded_kb,dropped_uploads\n');
for a = 1:nArms
    fprintf(fid, '%s,%.4f,%.4f,%.4f,%.1f,%.3f,%.1f\n', armNames{a}, Mm(a,1), Ms(a,1), Mm(a,3), Mm(a,4), Mm(a,5), Mm(a,6));
end
fclose(fid);
fprintf('\nfiles: %s, %s\n', fullfile(outDir, 'rounds.csv'), fullfile(outDir, 'summary.csv'));

results = struct('arms', {armNames}, 'mean', Mm, 'std', Ms, 'history', {firstHist});

% ---- plot ----------------------------------------------------------------
if o.plot && usejava('desktop')
    f = figure('Name', 'FL-AV prototype', 'Position', [100 100 1000 400]);
    subplot(1, 2, 1); hold on; grid on;
    for a = 1:nArms, plot(firstHist{a}(:,1), firstHist{a}(:,3), 'LineWidth', 1.3); end
    xlabel('round'); ylabel('global test accuracy'); title('Convergence');
    legend(armNames, 'Location', 'southeast', 'FontSize', 7);
    subplot(1, 2, 2); hold on; grid on;
    for a = 2:nArms, plot(firstHist{a}(:,1), cumsum(firstHist{a}(:,6)), 'LineWidth', 1.3); end
    xlabel('round'); ylabel('cumulative upload (KB)'); title('Communication');
    legend(armNames(2:end), 'Location', 'northwest', 'FontSize', 7);
    saveas(f, fullfile(outDir, 'accuracy.png'));
end
end

% ========================================================= 0. ARM OPTIONS ===
function op = arm_options(name)
op = struct('aware_select', false, 'quality_agg', false, 'topk', 1.0, 'dp_clip', 0, 'dp_noise', 0);
switch name
    case 'fedavg'
    case '+selection',    op.aware_select = true;
    case '+quality',      op.quality_agg = true;
    case 'proposed',      op.aware_select = true; op.quality_agg = true;
    case 'proposed+topk', op.aware_select = true; op.quality_agg = true; op.topk = 0.25;
    case 'proposed+dp',   op.aware_select = true; op.quality_agg = true; op.dp_clip = 1.0; op.dp_noise = 0.3;
end
end

% =============================================================== 1. DATA ====
function [clients, Xt, yt] = make_fleet(nClients, nClasses, dim, alpha, meanSamples)
% Private non-IID datasets per vehicle plus a clean global test set.
means = 0.6 * randn(nClasses, dim);       % small spread => overlapping classes => hard task
clients = repmat(struct('X', [], 'y', [], 'noise', 0, 'uptime', 0), nClients, 1);
for k = 1:nClients
    n = max(20, round(meanSamples * exp(0.4 * randn)));      % quantity skew
    pr = dirichlet(alpha * ones(1, nClasses));                % label skew
    y = sample_categorical(pr, n);
    noise = 0.8 + 1.7 * rand;                                 % sensor quality
    shift = 0.5 * randn(1, dim);                              % region / weather shift
    clients(k).X = means(y, :) + shift + noise * randn(n, dim);
    clients(k).y = y;
    clients(k).noise = noise;
    clients(k).uptime = 0.2 + 0.8 * rand;
end
yt = randi(nClasses, 1000, 1);
Xt = means(yt, :) + randn(1000, dim);
end

function describe_fleet(clients, nClasses)
fprintf('\nFLEET (private data never leaves a vehicle)\n');
fprintf('  %3s %7s %6s %6s  class histogram\n', 'veh', 'samples', 'noise', 'uptime');
for i = 1:numel(clients)
    c = clients(i);
    fprintf('  %3d %7d %6.2f %6.2f  [%s]\n', i - 1, numel(c.y), c.noise, c.uptime, ...
        strjoin(arrayfun(@num2str, accumarray(c.y, 1, [nClasses 1])', 'UniformOutput', false), ', '));
end
n = arrayfun(@(c) numel(c.y), clients);
fprintf('  total samples = %d, min/median/max per vehicle = %d/%d/%d\n', sum(n), min(n), round(median(n)), max(n));
end

function pr = dirichlet(a)
g = arrayfun(@rand_gamma, a);            % normalised Gamma draws
pr = g / sum(g);
end

function g = rand_gamma(a)
% Marsaglia-Tsang Gamma(a,1) sampler.
if a < 1
    g = rand_gamma(a + 1) * rand^(1 / a);
    return
end
d = a - 1/3; c = 1 / sqrt(9 * d);
while true
    x = randn; v = (1 + c * x)^3;
    if v > 0 && log(rand) < 0.5 * x^2 + d - d * v + d * log(v)
        g = d * v; return
    end
end
end

function y = sample_categorical(pr, n)
edges = cumsum(pr(:)'); edges(end) = 1;
y = zeros(n, 1);
for i = 1:n
    y(i) = find(rand <= edges, 1);
end
end

% ============================================================== 2. MODEL ====
function P = softmax(Z)
Z = Z - max(Z, [], 2);                    % subtract max for numerical stability
E = exp(Z);
P = E ./ sum(E, 2);
end

function [l, acc] = loss_acc(W, X, y)
% Cross-entropy loss and accuracy of the linear-softmax model W (dim x classes).
P = softmax(X * W);
l = mean(-log(P(sub2ind(size(P), (1:numel(y))', y)) + 1e-12));
[~, pred] = max(P, [], 2);
acc = mean(pred == y);
end

function W = local_train(W, X, y, epochs, lr, batch)
% Mini-batch SGD on a vehicle's private data, starting from the global weights.
n = numel(y);
for e = 1:epochs
    idx = randperm(n);
    for s = 1:batch:n
        b = idx(s:min(s + batch - 1, n));
        G = softmax(X(b, :) * W);
        lin = sub2ind(size(G), (1:numel(b))', y(b));
        G(lin) = G(lin) - 1;              % d(loss)/d(logits)
        W = W - lr * (X(b, :)' * G) / numel(b);
    end
end
end

function acc = per_vehicle_accuracy(W, clients)
acc = zeros(numel(clients), 1);
for i = 1:numel(clients)
    [~, acc(i)] = loss_acc(W, clients(i).X, clients(i).y);
end
end

% ========================================================= 3. FEDERATION ====
function sel = select_clients(clients, frac, uptimeMin, aware)
% Choose ceil(frac*N) vehicles (>=2). aware: only stable links eligible, sampled
% with probability proportional to uptime (weighted, without replacement).
N = numel(clients);
k = max(2, ceil(frac * N));
if ~aware
    sel = randperm(N, k);
    return
end
up = [clients.uptime];
pool = find(up >= uptimeMin);
if numel(pool) < k
    [~, order] = sort(up, 'descend');
    pool = order(1:k);
end
w = up(pool);
sel = zeros(1, k);
for i = 1:k
    j = find(rand <= cumsum(w / sum(w)), 1);
    sel(i) = pool(j); pool(j) = []; w(j) = [];
end
end

function d = clip_l2(d, clip)
% Bound one vehicle's influence: scale so that ||d||_2 <= clip.
d = d * min(1, clip / (norm(d(:)) + 1e-12));
end

function [sent, resid, nnz] = topk_sparsify(d, resid, ratio)
% Send only the largest `ratio` of entries; remember the rest (error feedback).
u = d + resid;
k = max(1, ceil(ratio * numel(u)));
[~, order] = sort(abs(u(:)), 'descend');
mask = false(size(u)); mask(order(1:k)) = true;
sent = zeros(size(u)); sent(mask) = u(mask);
resid = u - sent;
nnz = k;
end

function [W, hist] = run_fl(clients, Xt, yt, dim, nClasses, rounds, frac, seed, op, verbose, label)
% hist rows: [round, test_loss, test_acc, participants, dropped, uploaded_KB]
rng(seed);
W = zeros(dim, nClasses);
resid = repmat({zeros(dim, nClasses)}, 1, numel(clients));   % per-vehicle error feedback
hist = zeros(rounds, 6);
for r = 1:rounds
    sel = select_clients(clients, frac, 0.55, op.aware_select);
    stepSum = zeros(size(W)); wsum = 0; used = 0; dropped = 0; nbytes = 0;
    for i = sel
        c = clients(i);
        if rand > c.uptime, dropped = dropped + 1; continue; end   % V2I link lost
        Wk = local_train(W, c.X, c.y, 2, 0.05, 32);
        d = Wk - W;                                  % only the delta is transmitted
        if op.quality_agg
            q = 1 / (1 + loss_acc(Wk, c.X, c.y));    % poor local fit => smaller weight
        else
            q = 1;
        end
        if op.dp_clip > 0, d = clip_l2(d, op.dp_clip); end
        if op.topk < 1
            [d, resid{i}, nnz] = topk_sparsify(d, resid{i}, op.topk);
            nbytes = nbytes + nnz * (2 + 4);         % fp16 value + int32 index
        else
            nbytes = nbytes + numel(d) * 2;          % dense fp16
        end
        wt = q * numel(c.y);
        stepSum = stepSum + wt * d;
        wsum = wsum + wt; used = used + 1;
    end
    if used > 0
        step = stepSum / wsum;                       % a_k = q_k n_k / sum_j q_j n_j
        if op.dp_clip > 0 && op.dp_noise > 0
            step = step + (op.dp_noise * op.dp_clip / used) * randn(size(step));
        end
        W = W + step;
    end
    [l, a] = loss_acc(W, Xt, yt);
    hist(r, :) = [r, l, a, used, dropped, nbytes / 1e3];
    if verbose
        fprintf('  [%s] round %3d  selected=%d received=%d dropped=%d  test acc=%.3f loss=%.3f  up=%.1f KB\n', ...
            label, r, numel(sel), used, dropped, a, l, nbytes / 1e3);
    end
end
end

function [W, hist] = run_central(clients, Xt, yt, dim, nClasses, rounds, seed)
% Upper bound: pool all data on one server and train (not privacy preserving).
rng(seed);
X = vertcat(clients.X); y = vertcat(clients.y);
W = zeros(dim, nClasses);
hist = zeros(rounds, 6);
for r = 1:rounds
    W = local_train(W, X, y, 2, 0.05, 32);
    [l, a] = loss_acc(W, Xt, yt);
    hist(r, :) = [r, l, a, numel(clients), 0, 0];
end
end
