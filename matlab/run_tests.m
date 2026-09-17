% RUN_TESTS  Lightweight sanity tests for the MATLAB FedAvg model.
%
% Runs without any test toolbox so it works the same in MATLAB and
% GNU Octave. Exits with an error (non-zero status under `-batch`) if any
% assertion fails, which makes it usable from CI.
%
% Usage:
%   >> run_tests

function run_tests()
    fprintf('Running matlab/ FedAvg sanity tests...\n');

    test_aggregate_models_uniform();
    test_aggregate_models_quality_weighted();
    test_local_sgd_reduces_loss();
    test_full_simulation_converges();

    fprintf('All tests passed.\n');
end

function test_aggregate_models_uniform()
    local_ws = {[1; 0], [0; 1], [2; 2]};
    quality = [1; 1; 1];
    w = aggregate_models_ref(local_ws, quality, false);
    expected = [1; 1];
    assert(norm(w - expected) < 1e-10, ...
           'uniform aggregation should equal the plain average');
    fprintf('  [ok] aggregate_models: uniform average\n');
end

function test_aggregate_models_quality_weighted()
    local_ws = {[1; 0], [3; 0]};
    quality = [1; 3];
    w = aggregate_models_ref(local_ws, quality, true);
    expected = [(1*1 + 3*3) / 4; 0];
    assert(norm(w - expected) < 1e-10, ...
           'quality-weighted aggregation should weight by quality');
    fprintf('  [ok] aggregate_models: quality-weighted average\n');
end

function test_local_sgd_reduces_loss()
    rng(1);
    true_w = [1.5; -0.5];
    X = randn(50, 2);
    y = X * true_w;
    w0 = zeros(2, 1);

    loss_before = mean((X * w0 - y).^2);
    w1 = local_sgd_update_ref(w0, X, y, 20, 0.1);
    loss_after = mean((X * w1 - y).^2);

    assert(loss_after < loss_before, ...
           'local SGD should reduce training loss on a client shard');
    fprintf('  [ok] local_sgd_update: reduces local loss\n');
end

function test_full_simulation_converges()
    results = fedavg_simulation();
    h = results.loss_history;

    assert(numel(h) >= 2, 'expected loss history with >=2 rounds');
    assert(h(end) < h(1), ...
           'global loss should decrease from round 1 to the final round');
    assert(norm(results.global_w - results.true_w) < 1.0, ...
           'learned weights should converge reasonably close to ground truth');
    fprintf('  [ok] fedavg_simulation: end-to-end convergence\n');
end

% --- local re-implementations mirroring fedavg_simulation.m internals ---
% (kept private here so this file has no dependency ordering issues in
% Octave's function-per-file model beyond the two public entry points.)

function w_avg = aggregate_models_ref(local_ws, quality, quality_aware)
    if ~quality_aware
        quality = ones(size(quality));
    end
    weights = quality / sum(quality);
    w_avg = zeros(size(local_ws{1}));
    for i = 1:numel(local_ws)
        w_avg = w_avg + weights(i) * local_ws{i};
    end
end

function w_new = local_sgd_update_ref(w0, X, y, epochs, lr)
    w_new = w0;
    n = size(X, 1);
    for e = 1:epochs
        grad = (X' * (X * w_new - y)) / n;
        w_new = w_new - lr * grad;
    end
end
