% FEDAVG_SIMULATION  Working MATLAB model of the FL-AV problem design.
%
% Simulates the Federated Averaging (FedAvg) loop described in the
% project's PRD: a fleet of N simulated vehicles ("clients") each hold a
% private, non-IID shard of sensor data and jointly train one shared
% linear model without ever sharing raw data -- only local weight deltas
% are sent to a Road-Side Unit (RSU) that performs quality-aware
% aggregation.
%
% This is a compact, dependency-free (no toolboxes required) MATLAB
% model intended to validate the *problem design*: client partitioning,
% local training, client selection, quality-aware aggregation, and
% convergence tracking -- mirroring fedav/ in this repository, but as a
% single runnable MATLAB script.
%
% Usage (MATLAB or GNU Octave):
%   >> fedavg_simulation
%
% Run the unit tests with:
%   >> run_tests

function results = fedavg_simulation()

    rng(42); % reproducibility

    cfg = default_config();

    fprintf('FL-AV FedAvg simulation (MATLAB)\n');
    fprintf('  clients            : %d\n', cfg.num_clients);
    fprintf('  rounds             : %d\n', cfg.num_rounds);
    fprintf('  local epochs       : %d\n', cfg.local_epochs);
    fprintf('  selection fraction : %.2f\n', cfg.selection_fraction);
    fprintf('  quality-aware agg  : %d\n', cfg.quality_aware);
    fprintf('\n');

    % ---- Problem design: ground-truth model + non-IID client shards ----
    true_w = [2.5; -1.0; 0.75; 3.0; -2.0];
    d = numel(true_w);

    clients = make_clients(cfg.num_clients, d, true_w, cfg);

    % ---- Global model initialisation ----
    global_w = zeros(d, 1);

    history = zeros(cfg.num_rounds, 1);

    for r = 1:cfg.num_rounds
        selected = select_clients(cfg.num_clients, cfg.selection_fraction);

        local_ws = cell(numel(selected), 1);
        quality  = zeros(numel(selected), 1);

        for i = 1:numel(selected)
            c = clients(selected(i));
            local_ws{i} = local_sgd_update(global_w, c.X, c.y, ...
                                            cfg.local_epochs, cfg.lr);
            quality(i) = client_quality(c, cfg);
        end

        global_w = aggregate_models(local_ws, quality, cfg.quality_aware);

        history(r) = global_loss(global_w, clients);
        fprintf('round %2d/%2d  |  global loss = %.6f\n', ...
                r, cfg.num_rounds, history(r));
    end

    fprintf('\nfinal weight error ||w_hat - w*|| = %.6f\n', ...
            norm(global_w - true_w));

    results = struct();
    results.global_w = global_w;
    results.true_w = true_w;
    results.loss_history = history;
    results.config = cfg;

    if ~isdeployed() && usejava('desktop') == 0
        % headless (e.g. CI / Octave -q) : skip plotting, just report
    end
    try
        plot_convergence(history);
    catch err
        fprintf('(plotting skipped: %s)\n', err.message);
    end
end

function cfg = default_config()
    cfg.num_clients        = 20;
    cfg.samples_per_client = 40;
    cfg.num_rounds         = 40;
    cfg.local_epochs       = 5;
    cfg.lr                 = 0.01;
    cfg.selection_fraction = 0.4;   % C in the PRD
    cfg.noniid_skew        = 1.0;   % higher = more non-IID
    cfg.noise_std          = 0.3;
    cfg.quality_aware      = true;
end

function clients = make_clients(num_clients, d, true_w, cfg)
    clients = struct('X', {}, 'y', {}, 'n', {}, 'noise_std', {});
    for k = 1:num_clients
        % Non-IID partitioning: each client's features are drawn from a
        % shifted/scaled Gaussian, and label noise varies per client
        % (models heterogeneous sensor quality across the fleet).
        shift = cfg.noniid_skew * (mod(k, 5) - 2);
        scale = 0.5 + mod(k, 3) * 0.5;
        n = cfg.samples_per_client;

        X = scale * randn(n, d) + shift;
        client_noise = cfg.noise_std * (0.5 + rand());
        y = X * true_w + client_noise * randn(n, 1);

        clients(k).X = X;
        clients(k).y = y;
        clients(k).n = n;
        clients(k).noise_std = client_noise;
    end
end

function idx = select_clients(num_clients, fraction)
    m = max(1, round(fraction * num_clients));
    perm = randperm(num_clients);
    idx = perm(1:m);
end

function w_new = local_sgd_update(w0, X, y, epochs, lr)
    % Local training: plain gradient-descent on MSE loss, run for
    % `epochs` passes over the client's private shard.
    w_new = w0;
    n = size(X, 1);
    for e = 1:epochs
        grad = (X' * (X * w_new - y)) / n;
        w_new = w_new - lr * grad;
    end
end

function q = client_quality(c, cfg)
    % Quality-aware weighting (PRD q_k): larger, cleaner shards count
    % for more in the aggregation.
    if cfg.quality_aware
        q = c.n / (1 + c.noise_std);
    else
        q = c.n;
    end
end

function w_avg = aggregate_models(local_ws, quality, quality_aware)
    if ~quality_aware
        quality = ones(size(quality));
    end
    weights = quality / sum(quality);
    w_avg = zeros(size(local_ws{1}));
    for i = 1:numel(local_ws)
        w_avg = w_avg + weights(i) * local_ws{i};
    end
end

function loss = global_loss(w, clients)
    total = 0;
    n_total = 0;
    for k = 1:numel(clients)
        X = clients(k).X;
        y = clients(k).y;
        resid = X * w - y;
        total = total + sum(resid.^2);
        n_total = n_total + numel(y);
    end
    loss = total / n_total;
end

function plot_convergence(history)
    figure('Visible', 'off');
    plot(1:numel(history), history, '-o', 'LineWidth', 1.5);
    xlabel('Federated round');
    ylabel('Global MSE loss');
    title('FL-AV FedAvg convergence (MATLAB model)');
    grid on;
    out_dir = fileparts(mfilename('fullpath'));
    out_file = fullfile(out_dir, 'convergence.png');
    saveas(gcf, out_file);
    fprintf('convergence plot saved to %s\n', out_file);
end
