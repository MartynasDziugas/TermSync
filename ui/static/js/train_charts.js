/**
 * Standarto mokymo SVM / MLP palyginimo grafikai (Chart.js).
 * Naudojama /train (paskutinis snapshot) ir /train/job/... (po polling).
 */
(function (global) {
    function destroyChartList(list) {
        if (!list) return;
        list.forEach(function (c) {
            try {
                c.destroy();
            } catch (e) {}
        });
        list.length = 0;
    }

    function makeBarChart(chartList, canvasId, title, value, maxY) {
        var el = document.getElementById(canvasId);
        if (!el || typeof Chart === "undefined") return;
        var v = Number(value);
        if (!isFinite(v)) v = 0;
        var mx = maxY != null ? maxY : Math.max(1.05, v * 1.25, 0.01);
        var ch = new Chart(el, {
            type: "bar",
            data: {
                labels: [""],
                datasets: [
                    {
                        label: title,
                        data: [v],
                        backgroundColor: "rgba(236, 72, 153, 0.55)",
                        borderColor: "rgba(219, 39, 119, 0.9)",
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false },
                },
                scales: {
                    y: { beginAtZero: true, max: mx },
                },
            },
        });
        chartList.push(ch);
    }

    function makeLineChart(chartList, canvasId, title, xs, ys, yMax) {
        var el = document.getElementById(canvasId);
        if (!el || typeof Chart === "undefined") return;
        var labels = xs && xs.length ? xs.map(String) : ["1"];
        var data = ys && ys.length ? ys.map(Number) : [0];
        var mx = yMax != null ? yMax : null;
        var optsY = { beginAtZero: true };
        if (mx != null) optsY.max = mx;
        var ch = new Chart(el, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: title,
                        data: data,
                        borderColor: "rgba(219, 39, 119, 0.95)",
                        backgroundColor: "rgba(252, 231, 243, 0.5)",
                        tension: 0.15,
                        fill: false,
                        pointRadius: 2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false },
                },
                scales: {
                    x: { title: { display: true, text: "Epocha" } },
                    y: optsY,
                },
            },
        });
        chartList.push(ch);
    }

    /**
     * @param {HTMLElement} container
     * @param {{ result: object, active_model_type?: string, activeModelPostUrl?: string }} opts
     */
    function mountTrainCompare(container, opts) {
        if (!container || !opts || !opts.result) return;

        var prev = container._termSyncTrainCharts;
        if (prev) destroyChartList(prev);
        container._termSyncTrainCharts = [];

        var chartList = container._termSyncTrainCharts;
        var r = opts.result;
        var active = (opts.active_model_type || "svm").toLowerCase();
        var postUrl = opts.activeModelPostUrl || "";

        var svmC = r.svm_charts || {
            labels: ["Test"],
            accuracy: [r.svm_test_accuracy],
            f1: [r.svm_test_f1 != null ? r.svm_test_f1 : 0],
            loss: [r.svm_train_hinge != null ? r.svm_train_hinge : 0],
        };
        var mlpC = r.mlp_charts || {
            epochs: r.quick_mlp_epochs ? [String(r.quick_mlp_epochs)] : ["1"],
            accuracy: [r.mlp_test_accuracy],
            f1: [r.mlp_test_f1 != null ? r.mlp_test_f1 : 0],
            loss: [r.mlp_train_loss != null ? r.mlp_train_loss : 0],
        };

        var svmAcc =
            svmC.accuracy && svmC.accuracy.length
                ? svmC.accuracy[svmC.accuracy.length - 1]
                : r.svm_test_accuracy;
        var svmF1 =
            svmC.f1 && svmC.f1.length ? svmC.f1[svmC.f1.length - 1] : r.svm_test_f1 || 0;
        var svmLoss =
            svmC.loss && svmC.loss.length ? svmC.loss[svmC.loss.length - 1] : 0;

        var hasMlpSeries = mlpC.epochs && mlpC.epochs.length > 1;

        var modelPanel = postUrl
            ? '<div class="train-active-model-panel">' +
              '  <p class="train-active-badge" id="active-model-badge">Aktyvus modelis vertėjo peržiūrai: <strong id="active-model-label">' +
              (active === "mlp" ? "MLP" : "SVM") +
              "</strong></p>" +
              '  <div class="train-select-model-row">' +
              '    <label class="train-select-label">Pasirinkite klasifikatorių ' +
              '      <select id="active-model-select" aria-label="Modelis vertėjo peržiūrai">' +
              '        <option value="svm"' +
              (active !== "mlp" ? " selected" : "") +
              ">SVM</option>" +
              '        <option value="mlp"' +
              (active === "mlp" ? " selected" : "") +
              ">MLP</option>" +
              "      </select>" +
              "    </label>" +
              '    <button type="button" class="train-select-model-btn" id="btn-select-model">Select Model</button>' +
              "  </div>" +
              '  <p class="hint" id="select-model-status" role="status"></p>' +
              "</div>"
            : "";

        container.innerHTML =
            '<h2>Rezultatų palyginimas (test rinkinys)</h2>' +
            '<div class="train-compare-grid">' +
            '  <section class="train-model-card" aria-labelledby="svm-card-h">' +
            '    <h3 id="svm-card-h">SVM</h3>' +
            '    <div class="train-chart-grid">' +
            '      <div class="train-chart-wrap"><canvas id="chart-svm-acc" height="140"></canvas></div>' +
            '      <div class="train-chart-wrap"><canvas id="chart-svm-f1" height="140"></canvas></div>' +
            '      <div class="train-chart-wrap"><canvas id="chart-svm-loss" height="140"></canvas></div>' +
            "    </div>" +
            "  </section>" +
            '  <section class="train-model-card" aria-labelledby="mlp-card-h">' +
            '    <h3 id="mlp-card-h">MLP</h3>' +
            '    <div class="train-chart-grid">' +
            '      <div class="train-chart-wrap"><canvas id="chart-mlp-acc" height="140"></canvas></div>' +
            '      <div class="train-chart-wrap"><canvas id="chart-mlp-f1" height="140"></canvas></div>' +
            '      <div class="train-chart-wrap"><canvas id="chart-mlp-loss" height="140"></canvas></div>' +
            "    </div>" +
            "  </section>" +
            "</div>" +
            (modelPanel || "") +
            "<h2>Santrauka</h2>" +
            "<ul class=\"train-summary-list\">" +
            "<li>SVM accuracy (test): <strong>" +
            r.svm_test_accuracy +
            "</strong>" +
            (r.svm_test_f1 != null ? " · F1: <strong>" + r.svm_test_f1 + "</strong>" : "") +
            "</li>" +
            "<li>MLP accuracy (test): <strong>" +
            r.mlp_test_accuracy +
            "</strong>" +
            (r.mlp_test_f1 != null ? " · F1: <strong>" + r.mlp_test_f1 + "</strong>" : "") +
            " (" +
            r.quick_mlp_epochs +
            " epochų)</li>" +
            (r.mlp_train_loss != null
                ? "<li>MLP train loss (pabaigoje): <strong>" + r.mlp_train_loss + "</strong></li>"
                : "") +
            "<li>Standarto porų: " +
            r.n_aligned_pairs +
            "</li>" +
            "<li>Artefaktas: <code>" +
            (r.artifact_path || "") +
            "</code></li>" +
            "</ul>";

        makeBarChart(chartList, "chart-svm-acc", "Accuracy", svmAcc, 1.05);
        makeBarChart(chartList, "chart-svm-f1", "F1-score", svmF1, 1.05);
        makeBarChart(chartList, "chart-svm-loss", "Loss (hinge, train)", svmLoss, null);

        if (hasMlpSeries) {
            makeLineChart(chartList, "chart-mlp-acc", "Accuracy (test)", mlpC.epochs, mlpC.accuracy, 1.05);
            makeLineChart(chartList, "chart-mlp-f1", "F1-score (test)", mlpC.epochs, mlpC.f1, 1.05);
            makeLineChart(chartList, "chart-mlp-loss", "Loss (train CE)", mlpC.epochs, mlpC.loss, null);
        } else {
            makeBarChart(chartList, "chart-mlp-acc", "Accuracy", r.mlp_test_accuracy, 1.05);
            makeBarChart(
                chartList,
                "chart-mlp-f1",
                "F1-score",
                r.mlp_test_f1 != null ? r.mlp_test_f1 : 0,
                1.05
            );
            makeBarChart(
                chartList,
                "chart-mlp-loss",
                "Loss (train CE)",
                r.mlp_train_loss != null ? r.mlp_train_loss : 0,
                null
            );
        }

        if (postUrl) {
            var sel = document.getElementById("active-model-select");
            var badge = document.getElementById("active-model-label");
            var st = document.getElementById("select-model-status");
            var btn = document.getElementById("btn-select-model");
            if (btn) {
                btn.addEventListener("click", function () {
                    var v = sel && sel.value ? sel.value : "svm";
                    if (st) st.textContent = "Įrašoma…";
                    fetch(postUrl, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            Accept: "application/json",
                        },
                        body: JSON.stringify({ model_type: v }),
                    })
                        .then(function (res) {
                            return res.json().then(function (j) {
                                return { ok: res.ok, body: j };
                            });
                        })
                        .then(function (x) {
                            if (x.ok && x.body && x.body.ok) {
                                var t = (x.body.active_model_type || "svm").toLowerCase();
                                try {
                                    localStorage.setItem("termsync_active_model", t);
                                } catch (e) {}
                                if (badge) badge.textContent = t === "mlp" ? "MLP" : "SVM";
                                if (st) st.textContent = "Aktyvus modelis išsaugotas.";
                            } else {
                                if (st)
                                    st.textContent =
                                        x.body && x.body.error ? x.body.error : "Nepavyko išsaugoti.";
                            }
                        })
                        .catch(function (e) {
                            if (st) st.textContent = "Klaida: " + e;
                        });
                });
            }
        }
    }

    global.TermSyncTrainCharts = { mountTrainCompare: mountTrainCompare };
})(window);
