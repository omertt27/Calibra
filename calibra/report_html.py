"""
Visual HTML Report Generator.

Converts a DiagnosticReport into an interactive, visual dashboard page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from calibra.schema.report import DiagnosticReport, RiskLevel


def generate_html_report(
    report: DiagnosticReport, output_path: str, outliers: Optional[dict] = None
) -> None:
    """
    Generate a standalone HTML dashboard report and write it to `output_path`.
    """
    # Extract metrics for charts
    labels = [f"Ep {i}" for i in range(report.n_episodes)]

    # Safely get per-episode lists from raw_metrics
    def get_metric_values(key: str) -> list[float]:
        for res in report.analyzer_results:
            if key in res.raw_metrics:
                return [float(v) if v is not None else 0.0 for v in res.raw_metrics[key]]
        return []

    ldlj_vals = get_metric_values("per_episode_ldlj")
    vel_disc_vals = get_metric_values("per_episode_vel_disc_rate")
    ssl_novelty_vals = get_metric_values("per_episode_ssl_novelty")
    jitter_vals = get_metric_values("per_episode_jitter_cv")

    # Build flags list for JS
    flags_data = []
    for flag in report.flags:
        flags_data.append(
            {
                "level": flag.level.value,
                "metric": flag.metric,
                "observed": str(flag.observed),
                "threshold": flag.threshold,
                "interpretation": flag.interpretation,
                "implication": flag.implication,
            }
        )

    # Build outliers list
    outlier_list = []
    if outliers:
        for ep_idx, reasons in outliers.items():
            outlier_list.append(
                {
                    "index": ep_idx,
                    "episode_id": report.episode_ids[ep_idx]
                    if ep_idx < len(report.episode_ids)
                    else f"Ep {ep_idx}",
                    "reasons": reasons,
                }
            )

    html_template = """<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mergen Report — __DATASET_NAME__</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#eef2ff',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #0f172a;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 3px;
        }
    </style>
</head>
<body class="h-full flex flex-col font-sans">
    <!-- Header banner -->
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3">
            <div class="bg-indigo-600 text-white font-bold p-2.5 rounded-lg tracking-wider text-sm shadow-lg shadow-indigo-600/20">MERGEN</div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white">__DATASET_NAME__</h1>
                <p class="text-xs text-slate-400">Format: <span class="font-mono text-slate-300">__FORMAT__</span> &middot; Source: <span class="font-mono text-slate-300">__SOURCE_PATH__</span></p>
            </div>
        </div>
        <div class="flex gap-2">
            <div class="bg-slate-800 border border-slate-700/50 px-4 py-1.5 rounded-lg text-center">
                <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Episodes</div>
                <div class="text-lg font-bold text-white">__N_EPISODES__</div>
            </div>
            <div class="bg-slate-800 border border-slate-700/50 px-4 py-1.5 rounded-lg text-center">
                <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Samples</div>
                <div class="text-lg font-bold text-white">__N_SAMPLES__</div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 flex flex-col lg:flex-row overflow-hidden">
        <!-- Sidebar Navigation & Status Summary -->
        <section class="w-full lg:w-80 border-r border-slate-800 bg-slate-900/20 p-6 flex flex-col gap-6 overflow-y-auto">
            <div>
                <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Overall Diagnostics</h2>
                <div class="rounded-xl border border-slate-800 p-4 bg-slate-900/50 flex flex-col gap-4">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-slate-400">Critical Risks</span>
                        <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-red-950/50 text-red-400 border border-red-800/30">__CRITICAL_COUNT__</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-slate-400">Warnings</span>
                        <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-950/50 text-amber-400 border border-amber-800/30">__WARNING_COUNT__</span>
                    </div>
                    <div class="h-px bg-slate-800"></div>
                    <div class="flex flex-col gap-1">
                        <span class="text-xs text-slate-400">Target Policy</span>
                        <span class="text-sm font-semibold text-white">__POLICY_FAMILY__</span>
                    </div>
                </div>
            </div>

            <!-- Tabs -->
            <nav class="flex flex-col gap-1" id="nav-tabs">
                <button onclick="switchTab('overview')" id="btn-overview" class="tab-btn w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-white bg-indigo-600/10 border-l-2 border-indigo-500 text-left transition-all">
                    <span>Overview & Risks</span>
                </button>
                <button onclick="switchTab('charts')" id="btn-charts" class="tab-btn w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 text-left transition-all">
                    <span>Metric Distributions</span>
                </button>
                <button onclick="switchTab('remediation')" id="btn-remediation" class="tab-btn w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 text-left transition-all">
                    <span>Remediation Checklist</span>
                </button>
                <button onclick="switchTab('outliers')" id="btn-outliers" class="tab-btn w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 text-left transition-all">
                    <span>Outliers & Episodes</span>
                </button>
            </nav>
        </section>

        <!-- Content Area -->
        <section class="flex-1 p-6 overflow-y-auto custom-scrollbar bg-slate-950">
            <!-- TAB 1: OVERVIEW & RISKS -->
            <div id="tab-overview" class="tab-content flex flex-col gap-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Diagnostic Risk Findings</h2>
                    <p class="text-sm text-slate-400">Detailed list of issues identified across the temporal, smoothness, and dynamics analyzers.</p>
                </div>

                <div class="flex flex-col gap-3" id="risk-flags-container">
                    <!-- Loaded dynamically via JS -->
                </div>
            </div>

            <!-- TAB 2: CHARTS -->
            <div id="tab-charts" class="tab-content hidden flex flex-col gap-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Per-Episode Metric Distributions</h2>
                    <p class="text-sm text-slate-400">Interactive charts mapping values calculated for each individual demonstration.</p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900/40 border border-slate-800 p-4 rounded-xl">
                        <h3 class="text-sm font-semibold text-slate-300 mb-4">Log-Dimensionless Jerk (LDLJ) &middot; Higher is smoother</h3>
                        <div class="h-64"><canvas id="ldljChart"></canvas></div>
                    </div>
                    <div class="bg-slate-900/40 border border-slate-800 p-4 rounded-xl">
                        <h3 class="text-sm font-semibold text-slate-300 mb-4">Velocity Discontinuity Rate &middot; Lower is better</h3>
                        <div class="h-64"><canvas id="velChart"></canvas></div>
                    </div>
                    <div class="bg-slate-900/40 border border-slate-800 p-4 rounded-xl">
                        <h3 class="text-sm font-semibold text-slate-300 mb-4">SSL Trajectory Novelty (NN Distance)</h3>
                        <div class="h-64"><canvas id="sslChart"></canvas></div>
                    </div>
                    <div class="bg-slate-900/40 border border-slate-800 p-4 rounded-xl">
                        <h3 class="text-sm font-semibold text-slate-300 mb-4">Timestamp Jitter CV</h3>
                        <div class="h-64"><canvas id="jitterChart"></canvas></div>
                    </div>
                </div>
            </div>

            <!-- TAB 3: REMEDIATION CHECKLIST -->
            <div id="tab-remediation" class="tab-content hidden flex flex-col gap-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Actionable Remediation Roadmap</h2>
                    <p class="text-sm text-slate-400">Steps derived from analysis flags to resolve training dataset deficiencies.</p>
                </div>

                <div class="bg-slate-900/30 border border-slate-800 rounded-xl p-6 flex flex-col gap-4" id="checklist-container">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- TAB 4: OUTLIERS & EPISODES -->
            <div id="tab-outliers" class="tab-content hidden flex flex-col gap-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Episode Outlier Analysis</h2>
                    <p class="text-sm text-slate-400">Episodes flagged as anomalous via robust statistical outlier algorithms (MAD).</p>
                </div>

                <div class="border border-slate-800 rounded-xl bg-slate-900/20 overflow-hidden">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="bg-slate-900/80 border-b border-slate-800 text-slate-400">
                                <th class="px-6 py-3 font-semibold">Index</th>
                                <th class="px-6 py-3 font-semibold">Episode ID</th>
                                <th class="px-6 py-3 font-semibold">Anomalies Detected</th>
                            </tr>
                        </thead>
                        <tbody id="outliers-table-body" class="divide-y divide-slate-800/50">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <!-- Embedded Data & JS -->
    <script>
        const flags = __FLAGS_DATA__;
        const outliers = __OUTLIERS_DATA__;
        const labels = __LABELS__;
        
        const ldljVals = __LDLJ_VALS__;
        const velVals = __VEL_VALS__;
        const sslVals = __SSL_VALS__;
        const jitterVals = __JITTER_VALS__;

        // Tab switcher
        function switchTab(tabId) {
            // Hide all contents
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            // Remove active classes from buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('bg-indigo-600/10', 'border-l-2', 'border-indigo-500', 'text-white');
                btn.classList.add('text-slate-400');
            });

            // Show active content
            document.getElementById('tab-' + tabId).classList.remove('hidden');
            // Add active classes to target button
            const activeBtn = document.getElementById('btn-' + tabId);
            activeBtn.classList.add('bg-indigo-600/10', 'border-l-2', 'border-indigo-500', 'text-white');
            activeBtn.classList.remove('text-slate-400');
        }

        // Populate Risk Flags
        function renderFlags() {
            const container = document.getElementById('risk-flags-container');
            if (flags.length === 0) {
                container.innerHTML = `
                    <div class="border border-slate-800 rounded-xl p-8 text-center bg-slate-900/10">
                        <div class="text-3xl mb-2">🎉</div>
                        <h4 class="text-base font-bold text-white">No Risks Identified</h4>
                        <p class="text-xs text-slate-400 mt-1">This dataset complies with all diagnostic thresholds.</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = flags.map(f => {
                let colorClass = "border-blue-500/20 bg-blue-950/10 text-blue-400";
                let badgeClass = "bg-blue-950 border-blue-800 text-blue-400";
                if (f.level === "CRITICAL") {
                    colorClass = "border-red-500/20 bg-red-950/10 text-red-400";
                    badgeClass = "bg-red-950 border-red-800 text-red-400";
                } else if (f.level === "WARNING") {
                    colorClass = "border-amber-500/20 bg-amber-950/10 text-amber-400";
                    badgeClass = "bg-amber-950 border-amber-800 text-amber-400";
                }

                const thresholdSnippet = f.threshold ? `&middot; <span class="text-slate-400">Threshold:</span> <code class="font-mono text-white">${f.threshold}</code>` : '';

                return `
                    <div class="border rounded-xl p-5 flex flex-col gap-3 transition-all duration-300 hover:bg-slate-900/30 ${colorClass}">
                        <div class="flex items-center justify-between gap-3">
                            <h3 class="font-bold text-white text-base">${f.metric}</h3>
                            <span class="px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${badgeClass}">${f.level}</span>
                        </div>
                        <div class="text-sm text-slate-300">
                            <span class="text-slate-400">Observed Value:</span> <code class="font-mono text-white">${f.observed}</code>
                            ${thresholdSnippet}
                        </div>
                        <p class="text-sm text-slate-300 bg-slate-950/40 p-3 rounded-lg border border-slate-900">${f.interpretation}</p>
                        <p class="text-xs text-slate-400 border-l-2 border-slate-700 pl-3"><strong>Implication:</strong> ${f.implication}</p>
                    </div>
                `;
            }).join('');
        }

        // Populate Outliers Table
        function renderOutliers() {
            const tbody = document.getElementById('outliers-table-body');
            if (outliers.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="3" class="px-6 py-8 text-center text-slate-500">
                            No individual episode outliers flagged.
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = outliers.map(o => `
                <tr class="hover:bg-slate-900/20 transition-all">
                    <td class="px-6 py-4 font-mono font-semibold text-slate-400">${o.index}</td>
                    <td class="px-6 py-4 font-mono text-white">${o.episode_id}</td>
                    <td class="px-6 py-4 text-xs text-amber-400 font-medium">${o.reasons.join(', ')}</td>
                </tr>
            `).join('');
        }

        // Build Remediation Roadmap
        function renderChecklist() {
            const container = document.getElementById('checklist-container');
            if (flags.length === 0) {
                container.innerHTML = `
                    <div class="text-slate-400 text-sm">
                        No remediation actions required. The dataset is fully certified.
                    </div>
                `;
                return;
            }

            container.innerHTML = flags.map((f, i) => {
                let actionText = "";
                if (f.metric === "ldlj") {
                    actionText = "Apply trajectory smoothing (e.g. Savitzky-Golay filtering) to minimize action jerk before training.";
                } else if (f.metric === "velocity_discontinuity_rate") {
                    actionText = "Investigate packet drops, communication lag, or sudden joystick/teleop corrections in outlier episodes.";
                } else if (f.metric === "timestamp_jitter_cv" || f.metric === "timestamp_dropout_rate") {
                    actionText = "Verify dataset recording clocks. Interpolate missing timestamps or drop frames before feeding to policies.";
                } else if (f.metric === "ssl_trajectory_outliers") {
                    actionText = "Identify and prune the outlier episodes listed under the 'Outliers' tab using the `calibra prune` command.";
                } else if (f.metric === "contact_dropout") {
                    actionText = "Check contact sensor configuration mapping and calibration in the robot driver pipeline.";
                } else {
                    actionText = f.implication;
                }

                return `
                    <div class="flex gap-4 items-start bg-slate-900/20 border border-slate-900/60 p-4 rounded-xl hover:border-slate-800 transition-all">
                        <input type="checkbox" id="check-${i}" class="mt-1 h-4 w-4 rounded border-slate-800 text-indigo-600 focus:ring-indigo-600 bg-slate-950">
                        <label for="check-${i}" class="flex-1">
                            <span class="block text-sm font-semibold text-white">${f.metric} Fix</span>
                            <span class="block text-xs text-slate-400 mt-1">${actionText}</span>
                        </label>
                    </div>
                `;
            }).join('');
        }

        // Charts configuration
        function initCharts() {
            const chartDefaults = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(51, 65, 85, 0.1)' }, ticks: { color: '#64748b', font: { size: 10 } } }
                }
            };

            // LDLJ Chart
            if (ldljVals.length > 0) {
                new Chart(document.getElementById('ldljChart').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels.slice(0, ldljVals.length),
                        datasets: [{
                            data: ldljVals,
                            backgroundColor: '#6366f1',
                            borderRadius: 4
                        }]
                    },
                    options: chartDefaults
                });
            }

            // Velocity Chart
            if (velVals.length > 0) {
                new Chart(document.getElementById('velChart').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: labels.slice(0, velVals.length),
                        datasets: [{
                            data: velVals,
                            borderColor: '#fbbf24',
                            borderWidth: 2,
                            pointBackgroundColor: '#fbbf24',
                            tension: 0.15
                        }]
                    },
                    options: chartDefaults
                });
            }

            // SSL Chart
            if (sslVals.length > 0) {
                new Chart(document.getElementById('sslChart').getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels.slice(0, sslVals.length),
                        datasets: [{
                            data: sslVals,
                            backgroundColor: '#10b981',
                            borderRadius: 4
                        }]
                    },
                    options: chartDefaults
                });
            }

            // Jitter Chart
            if (jitterVals.length > 0) {
                new Chart(document.getElementById('jitterChart').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: labels.slice(0, jitterVals.length),
                        datasets: [{
                            data: jitterVals,
                            borderColor: '#ec4899',
                            borderWidth: 2,
                            tension: 0.15
                        }]
                    },
                    options: chartDefaults
                });
            }
        }

        // Initialize UI
        window.addEventListener('DOMContentLoaded', () => {
            renderFlags();
            renderOutliers();
            renderChecklist();
            initCharts();
        });
    </script>
</body>
</html>
"""

    # Do placeholders replacement
    html_content = (
        html_template.replace("__DATASET_NAME__", report.dataset_name)
        .replace("__FORMAT__", report.format)
        .replace("__SOURCE_PATH__", report.source_path)
        .replace("__N_EPISODES__", str(report.n_episodes))
        .replace("__N_SAMPLES__", f"{report.n_samples:,}")
        .replace("__CRITICAL_COUNT__", str(len(report.flags_at_level(RiskLevel.CRITICAL))))
        .replace("__WARNING_COUNT__", str(len(report.flags_at_level(RiskLevel.WARNING))))
        .replace("__POLICY_FAMILY__", report.policy_family or "None (Unspecified)")
        .replace("__FLAGS_DATA__", json.dumps(flags_data))
        .replace("__OUTLIERS_DATA__", json.dumps(outlier_list))
        .replace("__LABELS__", json.dumps(labels))
        .replace("__LDLJ_VALS__", json.dumps(ldlj_vals))
        .replace("__VEL_VALS__", json.dumps(vel_disc_vals))
        .replace("__SSL_VALS__", json.dumps(ssl_novelty_vals))
        .replace("__JITTER_VALS__", json.dumps(jitter_vals))
    )

    Path(output_path).write_text(html_content, encoding="utf-8")
    print(f"Visual HTML report generated successfully at: {output_path}")
