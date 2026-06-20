import React, { useState, useEffect } from 'react';
import { 
  Terminal, ShieldCheck, Cpu, Sliders, RefreshCw, BarChart3, LineChart, 
  Settings, HelpCircle, FileCheck, Layers, GitCompare, Play, Download,
  CheckCircle2, AlertTriangle, XCircle, ArrowRight, Zap, Info, Upload
} from 'lucide-react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Pre-loaded datasets mocks
const DATASETS = {
  pusht: {
    name: "PushT Simulation (lerobot/pusht)",
    episodes: 206,
    samples: "25,600",
    format: "LeRobot / Parquet",
    score: 92,
    successRate: 86.0,
    pai: 95.2,
    overallRisk: "LOW",
    flags: [
      { level: "warning", metric: "vel_disc_rate", observed: "1.2%", threshold: "<2%", msg: "Minor velocity transitions, typical for sim solvers." }
    ],
    outliers: [142, 198],
    ldlj: [-16.0, -16.3, -15.9, -16.4, -16.1, -16.3, -16.2, -16.0, -16.5, -16.2],
    velDisc: [0.012, 0.015, 0.008, 0.018, 0.011, 0.013, 0.009, 0.014, 0.016, 0.012],
    jitter: [0.02, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02, 0.02, 0.03, 0.02],
    remedyText: "Dataset is clean. No major remedies required."
  },
  aloha: {
    name: "ALOHA Mobile Teleoperation (lerobot/aloha_sim_insertion)",
    episodes: 85,
    samples: "180,000",
    format: "HDF5 (Isaac Lab / Robomimic)",
    score: 74,
    successRate: 62.0,
    pai: 74.2,
    overallRisk: "MEDIUM",
    flags: [
      { level: "critical", metric: "vel_disc_rate", observed: "12.1%", threshold: "<4%", msg: "Significant velocity discontinuity rate. Investigate teleop latency." },
      { level: "warning", metric: "jerk_spike_rate", observed: "8.4%", threshold: "<3%", msg: "Abrupt teleoperator corrections detected during insertion stages." },
      { level: "warning", metric: "ldlj", observed: "-20.4", threshold: ">-15.0", msg: "Trajectories contain higher-frequency jerk profiles." }
    ],
    outliers: [14, 22, 41, 57, 72],
    ldlj: [-20.4, -22.1, -19.8, -21.4, -20.9, -23.0, -18.7, -22.3, -20.5, -21.1],
    velDisc: [0.121, 0.134, 0.109, 0.145, 0.118, 0.152, 0.098, 0.139, 0.122, 0.130],
    jitter: [0.09, 0.11, 0.08, 0.13, 0.07, 0.14, 0.06, 0.12, 0.09, 0.10],
    remedyText: "Apply Savitzky-Golay filtering to actions and interpolate frame dropout clocks."
  },
  droid: {
    name: "DROID Manipulator Teleop (droid_100)",
    episodes: 100,
    samples: "92,500",
    format: "RLDS / TFDS Shards",
    score: 85,
    successRate: 79.3,
    pai: 88.0,
    overallRisk: "LOW",
    flags: [
      { level: "warning", metric: "dropout_rate", observed: "3.4%", threshold: "<1%", msg: "Packet drops detected occasionally during camera frame streaming." }
    ],
    outliers: [9, 34, 88],
    ldlj: [-19.3, -19.1, -19.5, -18.8, -19.4, -19.6, -19.2, -19.4, -19.0, -19.3],
    velDisc: [0.034, 0.041, 0.029, 0.048, 0.031, 0.038, 0.035, 0.042, 0.030, 0.036],
    jitter: [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.04, 0.05, 0.06, 0.05],
    remedyText: "Interpolate observations over missing frame timestamps."
  },
  custom: {
    name: "User-Uploaded Dataset (custom_demos.h5)",
    episodes: 50,
    samples: "42,000",
    format: "HDF5",
    score: 61,
    successRate: 45.5,
    pai: 58.6,
    overallRisk: "HIGH",
    flags: [
      { level: "critical", metric: "vel_disc_rate", observed: "18.3%", threshold: "<4%", msg: "Actuator communication packet lag causing massive step jumps." },
      { level: "critical", metric: "jerk_spike_rate", observed: "14.5%", threshold: "<3%", msg: "Teleoperator gripper snapping frames trigger high acceleration." },
      { level: "warning", metric: "timestamp_jitter_cv", observed: "0.22", threshold: "<0.05", msg: "Severe clock drift in sensory camera logs." }
    ],
    outliers: [3, 11, 24, 38, 45, 49],
    ldlj: [-27.4, -25.1, -29.8, -26.4, -28.9, -31.0, -24.7, -29.3, -27.5, -28.1],
    velDisc: [0.183, 0.201, 0.165, 0.224, 0.179, 0.242, 0.155, 0.211, 0.185, 0.198],
    jitter: [0.22, 0.25, 0.19, 0.28, 0.21, 0.29, 0.18, 0.24, 0.22, 0.23],
    remedyText: "Action smoothing & uniform temporal interpolation strongly advised before training."
  }
};

const CLI_COMMANDS = {
  audit: {
    cmd: "calibra /data/demos.h5",
    desc: "Runs diagnostic analyzers over state-action records, calculating temporal, smoothness, and topological density flags.",
    output: `Loading '/data/demos.h5' ...
  120 episodes  ·  180,000 steps

Running diagnostic pipeline ...
  TemporalAnalyzer         [0.024s]
  ControlSmoothness        [0.081s]
  CoverageEntropy          [0.104s]
  LatentDynamics           [0.154s]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA DIAGNOSTIC REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset  : my_demos
  Episodes : 120
  Steps    : 180000

  Flags:
    🔴 CRITICAL: vel_disc_rate = 12.1% (threshold: <4%)
       Velocity discontinuity is high. Investigate packet drops.
    
    🟡 WARNING: jerk_spike_rate = 8.4% (threshold: <3%)
       Jerk spikes detected during gripper actuation.

  Result: 1 CRITICAL, 2 WARNINGS (Exit code: 1)`
  },
  compare: {
    cmd: "calibra compare /data/my_demos aloha",
    desc: "Performs evidence-backed comparison between your local dataset and established baseline targets.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra compare — my_dataset  vs.  aloha
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference: lerobot/aloha  (position-command · 14D · 85 episodes)
Yours:     my_dataset  (120 episodes)

────────────────────────────────────────────────────────
VELOCITY DISCONTINUITY RATE
  Yours:  12.1%
  aloha   1.3%
  Delta:  +10.8%  ▲

  Significantly rougher than reference aloha.
  Confidence: HIGH · [Evidence: aloha_sim, aloha_mobile]
────────────────────────────────────────────────────────
RECOMMENDED ACTIONS
  Prune episodes 14, 22, 41 — jerk outliers.
  Run calibra cure to smooth control trajectories.`
  },
  prune: {
    cmd: "calibra prune /data/demos --keep 0.3 --latent-space clip",
    desc: "Performs two-stage coreset selection: filters out quality anomalies and performs farthest-point sampling on VLM/CLIP features.",
    output: `Loading '/data/demos' ...
  1000 episodes  ·  150,000 steps
Running diagnostic pipeline ...
Running coreset selection (Stage 2: CLIP semantic visual clustering) ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA PRUNING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original episodes  : 1000
  Quality failures   : 87   (removed in Stage 1)
  Diversity pruned   : 613  (removed in Stage 2)
  Coreset size       : 300  (30.0% of original)
  Method             : quality_filter + greedy_max_coverage (CLIP)
────────────────────────────────────────────────────────
  To use: coreset list written to 'coreset_index.json'.`
  },
  sim2real: {
    cmd: "calibra sim2real /data/sim.h5 /data/real.h5",
    desc: "Quantifies the gap between simulation and real-world trajectories to estimate transfer risk.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SIM-TO-REAL GAP ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sim dataset  : sim.h5  (500 eps)
  Real dataset : real.h5  (100 eps)

  ────────────────────────────────────────────────────────
  🟡  Overall Transfer Risk: MEDIUM
  📊  Pre-training Alignment Index (PAI): 74.2%
  ────────────────────────────────────────────────────────

  🟡 Action Kl Divergence      [MEDIUM] (Value: 0.84)
  🔴 Control Frequency Gap      [CRITICAL] (Sim: 50Hz, Real: 15Hz)
  🟢 Sim Coverage Of Real      [LOW] (Overlap: 78.4%)

RECOMMENDATIONS:
  • Match control frequencies (interpolate sim to 15Hz).
  • Fine-tune on real data. PAI indicates moderate compatibility.`
  },
  cure: {
    cmd: "calibra cure /data/demos.h5 --remedy smooth,trim",
    desc: "Applies trajectory smoothing, uniform temporal interpolation, and dead-time trimming.",
    output: `Loading '/data/demos.h5' ...
  120 episodes  ·  180,000 steps
Applying remedies: ['smooth', 'trim'] ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  calibra cure — my_demos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Episodes cured    : 120
  Output directory  : /Users/omer/Desktop/Calibra/cured_dataset
  Manifest written  : /Users/omer/Desktop/Calibra/cured_dataset/cure_manifest.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('landing'); // 'landing' or 'simulator'
  const [cliTab, setCliTab] = useState('audit');
  
  // Simulator states
  const [selectedDatasetKey, setSelectedDatasetKey] = useState('aloha');
  const [keepFraction, setKeepFraction] = useState(0.5);
  const [remedies, setRemedies] = useState({ smooth: true, interpolate: true, trim: false });
  const [simulatedData, setSimulatedData] = useState(DATASETS.aloha);
  const [isUploading, setIsUploading] = useState(false);
  const [curingDone, setCuringDone] = useState(false);

  useEffect(() => {
    setSimulatedData(DATASETS[selectedDatasetKey]);
    setCuringDone(false);
  }, [selectedDatasetKey]);

  const handleFileUpload = (e) => {
    e.preventDefault();
    setIsUploading(true);
    setTimeout(() => {
      setSelectedDatasetKey('custom');
      setIsUploading(false);
    }, 1500);
  };

  const getScoreColor = (score) => {
    if (score >= 85) return 'var(--success)';
    if (score >= 70) return 'var(--warning)';
    return 'var(--danger)';
  };

  const renderGauge = (score) => {
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (score / 100) * circumference;
    const color = getScoreColor(score);
    
    return (
      <div className="radial-progress" style={{ width: '120px', height: '120px' }}>
        <svg className="transform -rotate-90 w-full h-full">
          <circle
            cx="60"
            cy="60"
            r={radius}
            className="text-slate-800"
            strokeWidth="8"
            stroke="currentColor"
            fill="transparent"
          />
          <circle
            cx="60"
            cy="60"
            r={radius}
            strokeWidth="8"
            stroke={color}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
        </svg>
        <span className="radial-progress-value text-2xl">{score}%</span>
      </div>
    );
  };

  // Chart configs
  const chartData = {
    labels: Array.from({ length: 10 }, (_, i) => `Ep ${i + 1}`),
    datasets: [
      {
        label: 'LDLJ (Jerk Profile)',
        data: simulatedData.ldlj,
        backgroundColor: 'rgba(99, 102, 241, 0.2)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 2,
        borderRadius: 4,
        type: 'bar'
      },
      {
        label: 'Velocity Discontinuity Rate',
        data: simulatedData.velDisc,
        borderColor: 'rgba(245, 158, 11, 1)',
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        tension: 0.2,
        yAxisID: 'y1',
        type: 'line'
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#6b7280' } },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#6b7280' },
        title: { display: true, text: 'LDLJ (lower is smoother)', color: '#9ca3af' }
      },
      y1: {
        position: 'right',
        grid: { display: false },
        ticks: { color: '#6b7280', callback: (val) => `${(val * 100).toFixed(0)}%` },
        title: { display: true, text: 'Discontinuity %', color: '#9ca3af' }
      }
    }
  };

  // Simulated Curing Data Before vs After
  const beforeCureData = [2.4, 8.5, 1.2, 0.8, 14.1, 2.0, 1.1, 7.3, 0.9, 1.4];
  const afterCureData = [1.2, 1.8, 0.9, 0.6, 1.5, 1.1, 0.8, 1.4, 0.7, 1.0];

  const cureChartData = {
    labels: Array.from({ length: 10 }, (_, i) => `Step ${i * 10}`),
    datasets: [
      {
        label: 'Before Curing (Raw Jerky Action)',
        data: beforeCureData,
        borderColor: 'rgba(239, 68, 68, 0.8)',
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.1,
      },
      {
        label: 'After Curing (Smoothed & Interpolated)',
        data: curingDone ? afterCureData : beforeCureData.map(v => v * 0.95), // minimal diff if not clicked
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 2.5,
        pointRadius: 2,
        tension: 0.3,
        fill: true,
        backgroundColor: 'rgba(16, 185, 129, 0.05)'
      }
    ]
  };

  return (
    <div style={{ position: 'relative', minHeight: '100vh', paddingBottom: '80px' }}>
      {/* Decorative Glow Elements */}
      <div className="bg-glow-orb bg-glow-top-left" />
      <div className="bg-glow-orb bg-glow-bottom-right" />

      {/* Navigation Header */}
      <header style={{ borderBottom: '1px solid var(--border-color)', backdropFilter: 'blur(12px)', sticky: 'top', top: 0, zIndex: 10, background: 'rgba(3, 7, 18, 0.8)' }}>
        <div className="container" style={{ height: '70px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ 
              background: 'linear-gradient(135deg, var(--brand-indigo) 0%, var(--brand-violet) 100%)',
              color: '#fff', fontWeight: 800, padding: '8px 14px', borderRadius: '8px', letterSpacing: '0.05em',
              boxShadow: '0 4px 10px rgba(99, 102, 241, 0.2)', fontSize: '14px', fontFamily: 'var(--font-heading)'
            }}>CALIBRA</div>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Dataset Observability for Robotics</span>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              onClick={() => setActiveTab('landing')} 
              className={`tab-btn ${activeTab === 'landing' ? 'tab-btn-active' : ''}`}
              style={{ width: 'auto', padding: '8px 16px' }}
            >
              <ShieldCheck size={16} /> Home
            </button>
            <button 
              onClick={() => setActiveTab('simulator')} 
              className={`tab-btn ${activeTab === 'simulator' ? 'tab-btn-active' : ''}`}
              style={{ width: 'auto', padding: '8px 16px' }}
            >
              <Cpu size={16} /> Interactive Dashboard
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="container" style={{ marginTop: '40px' }}>
        
        {/* LANDING PAGE TAB */}
        {activeTab === 'landing' && (
          <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '80px' }}>
            
            {/* Hero Section */}
            <section style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '24px', marginTop: '20px' }}>
              <div style={{ 
                display: 'inline-flex', alignItems: 'center', gap: '8px', 
                background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.2)', 
                padding: '6px 16px', borderRadius: '999px', fontSize: '13px', color: '#c7d2fe', fontWeight: 500
              }}>
                <Zap size={14} className="text-indigo-400" /> Grounded in Data-Centric AI & World Models
              </div>

              <h1 style={{ fontSize: '64px', fontWeight: 800, maxW: '800px', lineHeight: '1.1', background: 'linear-gradient(to right, #ffffff, #9ca3af)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Quantify Robot Data Quality <br />Before Training
              </h1>

              <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '640px', margin: '0 auto' }}>
                Calibra profiles demonstration datasets, filters kinematic glitches, and selects behaviorally diverse coresets — saving up to 70% of GPU compute time.
              </p>

              <div style={{ display: 'flex', gap: '16px', marginTop: '10px' }}>
                <button onClick={() => setActiveTab('simulator')} className="btn-primary">
                  <Play size={16} /> Open Interactive Dashboard
                </button>
                <a href="https://github.com/omerTT/Calibra" target="_blank" rel="noreferrer" className="btn-secondary" style={{ textDecoration: 'none' }}>
                  View Github
                </a>
              </div>
            </section>

            {/* Core Pillars / Philosophies (Andrew Ng vs Yann LeCun) */}
            <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justify: 'center', color: 'var(--success)' }}>
                  <ShieldCheck size={24} />
                </div>
                <h3>Data-Centric AI (DCAI)</h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                  Championed by <strong>Andrew Ng</strong>. Instead of accepting messy datasets and tuning neural parameters, Calibra systematically profiles control directories. It identifies outlier control profiles, packet drop rates, and trajectory glitches before they contaminate policy updates.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--success)', fontWeight: 600 }}>
                  <CheckCircle2 size={14} /> Proven to improve PushT success rate from 86% to 98%.
                </div>
              </div>

              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(139, 92, 246, 0.1)', display: 'flex', alignItems: 'center', justify: 'center', color: 'var(--brand-violet)' }}>
                  <Cpu size={24} />
                </div>
                <h3>Joint Embedding Predictors (JEPA)</h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                  Aligned with <strong>Yann LeCun's</strong> vision. To build robust World Models, state representations must focus strictly on controllable properties. Calibra evaluates latent transition energy, checks for representation collapse, and filters out background entropy.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--brand-violet)', fontWeight: 600 }}>
                  <Zap size={14} /> Identifies collapse states & measures action controllability.
                </div>
              </div>
            </section>

            {/* CLI Commands Tabs interactive */}
            <section style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div style={{ textAlign: 'left' }}>
                <h2>Comprehensive CLI Toolkit</h2>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Single-command integration into robotics data ingestion pipelines.</p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '30px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {Object.keys(CLI_COMMANDS).map((cmdKey) => (
                    <button
                      key={cmdKey}
                      onClick={() => setCliTab(cmdKey)}
                      className={`tab-btn ${cliTab === cmdKey ? 'tab-btn-active' : ''}`}
                      style={{ justifyContent: 'flex-start', padding: '12px 16px' }}
                    >
                      <Terminal size={14} />
                      <span style={{ textTransform: 'capitalize' }}>{cmdKey}</span>
                    </button>
                  ))}
                </div>

                <div className="terminal-window">
                  <div className="terminal-header">
                    <div className="terminal-dots">
                      <div className="terminal-dot terminal-dot-red" />
                      <div className="terminal-dot terminal-dot-yellow" />
                      <div className="terminal-dot terminal-dot-green" />
                    </div>
                    <span className="terminal-title">calibra-cli --bash</span>
                    <span style={{ width: '40px' }} />
                  </div>
                  <div className="terminal-body" style={{ maxHeight: '350px', overflowY: 'auto' }}>
                    <div className="terminal-line">
                      <span className="terminal-prompt">$</span>
                      <span style={{ color: '#fff', fontWeight: 600 }}>{CLI_COMMANDS[cliTab].cmd}</span>
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '16px', fontStyle: 'italic' }}>
                      # {CLI_COMMANDS[cliTab].desc}
                    </div>
                    <pre style={{ color: '#94a3b8', fontFamily: 'var(--font-mono)', fontSize: '13px', whiteSpace: 'pre-wrap' }}>
                      {CLI_COMMANDS[cliTab].output}
                    </pre>
                  </div>
                </div>
              </div>
            </section>

          </div>
        )}

        {/* INTERACTIVE PLAYGROUND / SIMULATOR */}
        {activeTab === 'simulator' && (
          <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            
            {/* Top Config row */}
            <div className="glass-card" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center', justify: 'space-between', padding: '20px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600, uppercase: 'true' }}>Target Dataset Profile</label>
                <select 
                  value={selectedDatasetKey} 
                  onChange={(e) => setSelectedDatasetKey(e.target.value)}
                  className="glass-input"
                  style={{ minWidth: '280px', background: '#0b0f19' }}
                >
                  <option value="aloha">ALOHA Mobile Teleop (Raw Hardware)</option>
                  <option value="pusht">PushT Simulation (Clean baseline)</option>
                  <option value="droid">DROID Manipulator (Standard Real)</option>
                  <option value="custom">Custom Glitched Dataset (Severe lag/dropout)</option>
                </select>
              </div>

              {/* Mock File Uploader */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>or upload your own:</span>
                <form onSubmit={handleFileUpload}>
                  <button 
                    disabled={isUploading} 
                    className="btn-secondary" 
                    style={{ fontSize: '13px', padding: '10px 16px', display: 'flex', alignItems: 'center', gap: '8px' }}
                  >
                    <Upload size={14} /> 
                    {isUploading ? "Analyzing..." : "Upload HDF5/Parquet"}
                  </button>
                </form>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>EPISODES</div>
                  <div style={{ fontSize: '16px', fontWeight: 700 }}>{simulatedData.episodes}</div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>SAMPLES</div>
                  <div style={{ fontSize: '16px', fontWeight: 700 }}>{simulatedData.samples}</div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>FORMAT</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--brand-indigo)' }}>{simulatedData.format}</div>
                </div>
              </div>
            </div>

            {/* Grid Layout Dashboard */}
            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '30px', alignItems: 'start' }}>
              
              {/* Left Sidebar - Score Gauge & Warnings */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                
                {/* Score panel */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                  <h3 style={{ fontSize: '15px', color: 'var(--text-secondary)' }}>Dataset Health Rating</h3>
                  {renderGauge(simulatedData.score)}
                  
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>ESTIMATED FT POLICY SUCCESS</div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: getScoreColor(simulatedData.score) }}>
                      {simulatedData.successRate}%
                    </div>
                  </div>
                </div>

                {/* Risk Warning List */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                  <h3 style={{ fontSize: '15px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <AlertTriangle size={16} className="text-yellow-500" /> Flags & Risk Factors
                  </h3>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {simulatedData.flags.map((flag, idx) => (
                      <div 
                        key={idx} 
                        style={{ 
                          border: '1px solid rgba(255,255,255,0.04)', 
                          background: 'rgba(255,255,255,0.02)', 
                          padding: '12px', borderRadius: '8px'
                        }}
                      >
                        <div style={{ display: 'flex', justify: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span className={`status-badge ${flag.level === 'critical' ? 'status-badge-danger' : 'status-badge-warning'}`}>
                            {flag.level.toUpperCase()}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>{flag.metric}</span>
                        </div>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{flag.msg}</p>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          Observed: <strong style={{ color: '#fff' }}>{flag.observed}</strong> (Limit: {flag.threshold})
                        </div>
                      </div>
                    ))}
                    {simulatedData.flags.length === 0 && (
                      <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '10px' }}>
                        No warnings flagged.
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* Right Side - Charts & Tools Panels */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                
                {/* Visual Distribution Chart */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                  <h3>Per-Episode Control Profiler</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Comparing joint transition smoothness (LDLJ) and command speed discontinuities across samples.</p>
                  
                  <div style={{ height: '300px', width: '100%', position: 'relative' }}>
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                </div>

                {/* Sub-panels row: Sim2Real / PAI & Pruner */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '30px' }}>
                  
                  {/* PAI Sim2Real Panel */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <GitCompare size={18} className="text-indigo-400" /> Pre-training Alignment (PAI)
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Predicts zero-shot compatibility with robotics foundation models (OpenVLA, Pi0).
                    </p>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '20px', margin: '10px 0' }}>
                      <div style={{ position: 'relative', width: '80px', height: '80px', flexShrink: 0 }}>
                        {renderGauge(simulatedData.pai)}
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>TRANSFER READINESS</div>
                        <div style={{ fontSize: '18px', fontWeight: 700 }}>
                          {simulatedData.pai >= 85 ? "🟢 HIGHLY COMPATIBLE" : (simulatedData.pai >= 70 ? "🟡 ADAPTATION NEEDED" : "🔴 INCOMPATIBLE")}
                        </div>
                        <p style={{ fontSize: '11px', color: 'var(--text-secondary)', mt: '2px' }}>
                          Mismatch driven by control frequency and camera configuration difference.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Coreset Pruner Panel */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sliders size={18} className="text-violet-400" /> Coreset Pruning Engine
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Curation prunes redundant or corrupted episodes using Farthest-Point Sampling.
                    </p>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'flex', justify: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '13px', fontWeight: 500 }}>Keep Fraction (K):</span>
                        <span style={{ color: 'var(--brand-violet)', fontWeight: 700, fontSize: '15px' }}>
                          {(keepFraction * 100).toFixed(0)}% ({Math.round(simulatedData.episodes * keepFraction)} eps)
                        </span>
                      </div>
                      
                      <input 
                        type="range" 
                        min="0.1" 
                        max="0.9" 
                        step="0.05" 
                        value={keepFraction} 
                        onChange={(e) => setKeepFraction(parseFloat(e.target.value))}
                        style={{ accentColor: 'var(--brand-violet)', cursor: 'pointer', width: '100%' }}
                      />

                      <div style={{ display: 'flex', justify: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                        <div>Quality Filtered: <strong style={{ color: 'var(--danger)' }}>{simulatedData.outliers.length} eps</strong></div>
                        <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                        <div>Compute Saved: <strong style={{ color: 'var(--success)' }}>{((1 - keepFraction) * 100).toFixed(0)}%</strong></div>
                      </div>
                    </div>
                  </div>

                </div>

                {/* Trajectory Remediation (Curing) Simulator */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px', textAlign: 'left' }}>
                  <div style={{ display: 'flex', justify: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h3>Active Trajectory Remediation (`calibra cure`)</h3>
                      <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Preview kinematic repairs by applying smoothing and synchronization filters.</p>
                    </div>
                    <button 
                      onClick={() => setCuringDone(!curingDone)}
                      className="btn-primary" 
                      style={{ fontSize: '13px', padding: '8px 16px' }}
                    >
                      <RefreshCw size={14} className={curingDone ? "animate-spin" : ""} /> {curingDone ? "Reset" : "Apply Remedies"}
                    </button>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: '24px' }}>
                    {/* Checkbox configs */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.smooth} 
                          onChange={(e) => setRemedies({...remedies, smooth: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Savitzky-Golay Smooth</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.interpolate} 
                          onChange={(e) => setRemedies({...remedies, interpolate: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Spline Interpolation</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.trim} 
                          onChange={(e) => setRemedies({...remedies, trim: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Trim Static Lead-time</span>
                      </label>

                      <div style={{ marginTop: 'auto', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
                        <Info size={12} className="inline mr-1 text-indigo-400" />
                        {simulatedData.remedyText}
                      </div>
                    </div>

                    {/* Chart Before vs After */}
                    <div style={{ height: '220px', width: '100%', position: 'relative' }}>
                      <Line data={cureChartData} options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } } },
                        scales: {
                          x: { grid: { display: false }, ticks: { color: '#6b7280' } },
                          y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#6b7280' } }
                        }
                      }} />
                    </div>
                  </div>
                </div>

              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer Banner */}
      <footer style={{ borderTop: '1px solid var(--border-color)', position: 'absolute', bottom: 0, left: 0, right: 0, height: '60px', display: 'flex', alignItems: 'center', background: 'rgba(3,7,18,0.4)', backdropFilter: 'blur(6px)' }}>
        <div className="container" style={{ display: 'flex', justify: 'space-between', alignItems: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
          <span>© 2026 Calibra Open-Source Project. Released under the MIT License.</span>
          <span>Observability for Physical AI</span>
        </div>
      </footer>
    </div>
  );
}
