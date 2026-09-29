function results = fl_prototype(nClients, nRounds, seed)
%FL_PROTOTYPE Minimal federated-learning prototype for collaborative AV training.
%   A fleet of vehicles trains a shared softmax classifier (stand-in for a
%   perception model) on private, non-IID, quality-varying data. Only weight
%   deltas reach the RSU. Mirrors python_prototype/fl_prototype.py.
%
%   Arms: centralized (upper bound) | FedAvg | proposed (connectivity-aware
%   selection + quality-aware aggregation).
%
%   results = fl_prototype()                 % defaults: 20 vehicles, 30 rounds
%   results = fl_prototype(20, 30, 1)
%   Uses base MATLAB only (no toolboxes).

if nargin < 1, nClients = 20; end
if nargin < 2, nRounds  = 30; end
if nargin < 3, seed     = 1;  end
nClasses = 6; dim = 16; alpha = 0.4; frac = 0.3;

rng(seed);
[clients, Xt, yt] = make_fleet(nClients, nClasses, dim, alpha, 150);

hC = run_central(clients, Xt, yt, dim, nClasses, nRounds, seed);
hF = run_fl(clients, Xt, yt, dim, nClasses, nRounds, frac, false, seed);
hP = run_fl(clients, Xt, yt, dim, nClasses, nRounds, frac, true,  seed);

results = struct('centralized', hC, 'fedavg', hF, 'proposed', hP);

fprintf('%d vehicles, %d rounds, Dirichlet alpha=%.1f\n', nClients, nRounds, alpha);
fprintf('  %-12s final acc = %.3f   final loss = %.3f\n', 'centralized', hC(end,3), hC(end,2));
fprintf('  %-12s final acc = %.3f   final loss = %.3f\n', 'fedavg',      hF(end,3), hF(end,2));
fprintf('  %-12s final acc = %.3f   final loss = %.3f\n', 'proposed',    hP(end,3), hP(end,2));

outDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outDir, 'dir'), mkdir(outDir); end
names = {'centralized','fedavg','proposed'}; H = {hC, hF, hP};
fid = fopen(fullfile(outDir, 'rounds.csv'), 'w');
fprintf(fid, 'arm,round,test_loss,test_acc,participants\n');
for a = 1:3
    for r = 1:size(H{a},1)
        fprintf(fid, '%s,%d,%.6f,%.6f,%d\n', names{a}, H{a}(r,1), H{a}(r,2), H{a}(r,3), H{a}(r,4));
    end
end
fclose(fid);

% Convergence plot (skipped on headless sessions without a display).
if usejava('desktop')
    figure('Name', 'FL-AV prototype'); hold on; grid on;
    plot(hC(:,1), hC(:,3), 'k--', 'LineWidth', 1.5);
    plot(hF(:,1), hF(:,3), 'b-',  'LineWidth', 1.5);
    plot(hP(:,1), hP(:,3), 'r-',  'LineWidth', 1.5);
    xlabel('round'); ylabel('global test accuracy');
    legend(names, 'Location', 'southeast'); title('Federated vs centralized');
    saveas(gcf, fullfile(outDir, 'accuracy.png'));
end
end

% ---------------------------------------------------------------- data ----
function [clients, Xt, yt] = make_fleet(nClients, nClasses, dim, alpha, samples)
% Non-IID: Dirichlet label skew, log-normal quantity skew, per-vehicle
% feature shift (region) and noise (sensor quality).
means = 0.6 * randn(nClasses, dim);
clients = repmat(struct('X', [], 'y', [], 'uptime', 0), nClients, 1);
for k = 1:nClients
    n = max(20, round(samples * exp(0.4 * randn)));
    p = dirichlet(alpha * ones(1, nClasses));
    y = sample_categorical(p, n);
    noise = 0.8 + 1.7 * rand;                       % sensor degradation
    shift = 0.5 * randn(1, dim);                    % region / weather shift
    clients(k).X = means(y, :) + shift + noise * randn(n, dim);
    clients(k).y = y;
    clients(k).uptime = 0.2 + 0.8 * rand;
end
yt = randi(nClasses, 1000, 1);
Xt = means(yt, :) + randn(1000, dim);               % clean global test set
end

function p = dirichlet(a)
% Dirichlet sample via normalised Gamma draws (Gamma via Marsaglia-Tsang).
g = arrayfun(@rand_gamma, a);
p = g / sum(g);
end

function g = rand_gamma(a)
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

function y = sample_categorical(p, n)
edges = cumsum(p(:)');
edges(end) = 1;
y = zeros(n, 1);
for i = 1:n
    y(i) = find(rand <= edges, 1);
end
end

% --------------------------------------------------------------- model ----
function P = softmax(Z)
Z = Z - max(Z, [], 2);
E = exp(Z);
P = E ./ sum(E, 2);
end

function [l, acc] = loss_acc(W, X, y)
P = softmax(X * W);
idx = sub2ind(size(P), (1:numel(y))', y);
l = mean(-log(P(idx) + 1e-12));
[~, pred] = max(P, [], 2);
acc = mean(pred == y);
end

function W = local_train(W, X, y, epochs, lr, batch)
n = numel(y);
for e = 1:epochs
    idx = randperm(n);
    for s = 1:batch:n
        b = idx(s:min(s + batch - 1, n));
        P = softmax(X(b, :) * W);
        P(sub2ind(size(P), (1:numel(b))', y(b))) = P(sub2ind(size(P), (1:numel(b))', y(b))) - 1;
        W = W - lr * (X(b, :)' * P) / numel(b);
    end
end
end

% ---------------------------------------------------------- federation ----
function sel = select_clients(clients, frac, uptimeMin, aware)
N = numel(clients);
k = max(2, ceil(frac * N));
if ~aware
    sel = randperm(N, k);
    return
end
up = [clients.uptime];
ok = find(up >= uptimeMin);
if numel(ok) < k
    [~, order] = sort(up, 'descend');
    ok = order(1:k);
end
% Weighted sampling without replacement, probability proportional to uptime.
sel = zeros(1, k); pool = ok; w = up(ok);
for i = 1:k
    j = find(rand <= cumsum(w / sum(w)), 1);
    sel(i) = pool(j); pool(j) = []; w(j) = [];
end
end

function hist = run_fl(clients, Xt, yt, dim, nClasses, rounds, frac, aware, seed)
rng(seed);
W = zeros(dim, nClasses);
hist = zeros(rounds, 4);
for r = 1:rounds
    sel = select_clients(clients, frac, 0.55, aware);
    dsum = zeros(size(W)); wsum = 0; used = 0;
    for i = sel
        c = clients(i);
        if rand > c.uptime, continue; end            % V2I dropout
        Wk = local_train(W, c.X, c.y, 2, 0.05, 32);
        if aware
            q = 1 / (1 + loss_acc(Wk, c.X, c.y));
        else
            q = 1;
        end
        wt = q * numel(c.y);
        dsum = dsum + wt * (Wk - W);
        wsum = wsum + wt; used = used + 1;
    end
    if used > 0, W = W + dsum / wsum; end
    [l, a] = loss_acc(W, Xt, yt);
    hist(r, :) = [r, l, a, used];
end
end

function hist = run_central(clients, Xt, yt, dim, nClasses, rounds, seed)
rng(seed);
X = vertcat(clients.X); y = vertcat(clients.y);
W = zeros(dim, nClasses);
hist = zeros(rounds, 4);
for r = 1:rounds
    W = local_train(W, X, y, 2, 0.05, 32);
    [l, a] = loss_acc(W, Xt, yt);
    hist(r, :) = [r, l, a, numel(clients)];
end
end
