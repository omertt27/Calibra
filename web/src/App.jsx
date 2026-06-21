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

// Pre-loaded SRE incident domains and patterns mock
const INCIDENT_DOMAINS = {
  db_pool: {
    name: "Postgres Connection Pool Exhaustion (pg-pool-exhaustion)",
    incidents: 85,
    patternsLearned: 8,
    format: "pg_stat_activity logs / PGBouncer metrics",
    score: 72,
    successRate: 64.0,
    pai: 74.2,
    overallRisk: "MEDIUM",
    flags: [
      { level: "critical", metric: "User Override rate", observed: "12.1%", threshold: "<5%", msg: "Operator issued database failover command, overriding connection recycling." },
      { level: "warning", metric: "Low-Confidence Escalations", observed: "8.4%", threshold: "<5%", msg: "High latency variance in connection acquisition. Agent escalated to Slack." },
      { level: "warning", metric: "False-Positives", observed: "6.4%", threshold: "<2%", msg: "Pool scaling triggered during routine database backup script." }
    ],
    outliers: [14, 22, 41, 57, 72],
    cpuBefore: [65.4, 72.1, 63.8, 69.4, 67.9, 75.0, 61.7, 72.3, 66.5, 68.1],
    latencyBefore: [0.121, 0.134, 0.109, 0.145, 0.118, 0.152, 0.098, 0.139, 0.122, 0.130],
    jitter: [0.09, 0.11, 0.08, 0.13, 0.07, 0.14, 0.06, 0.12, 0.09, 0.10],
    remedyText: "Configure database query timeouts and increase pool recycling limits."
  },
  k8s_crash: {
    name: "Kubernetes Pod CrashLoopBackOff (k8s-pod-restart)",
    incidents: 147,
    patternsLearned: 14,
    format: "Kubernetes Events / Prometheus Telemetry",
    score: 94,
    successRate: 88.0,
    pai: 95.2,
    overallRisk: "LOW",
    flags: [
      { level: "warning", metric: "Manual Override rate", observed: "1.2%", threshold: "<5%", msg: "Operator manually rolled back replica set scale command." }
    ],
    outliers: [102, 142],
    cpuBefore: [45.0, 48.2, 42.1, 46.4, 44.9, 47.1, 43.5, 45.8, 48.0, 44.2],
    latencyBefore: [0.012, 0.015, 0.008, 0.018, 0.011, 0.013, 0.009, 0.014, 0.016, 0.012],
    jitter: [0.02, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02, 0.02, 0.03, 0.02],
    remedyText: "Patterns fully compiled. Action safe for autonomous execution."
  },
  redis_evict: {
    name: "Redis Memory Eviction Storm (redis-memory-evictions)",
    incidents: 62,
    patternsLearned: 11,
    format: "Redis Info stats / Sentinel alerts",
    score: 81,
    successRate: 74.0,
    pai: 88.0,
    overallRisk: "LOW",
    flags: [
      { level: "warning", metric: "False-Positives", observed: "3.4%", threshold: "<2%", msg: "Transient spike in cache keys flagged as cache memory leak." }
    ],
    outliers: [9, 34, 88],
    cpuBefore: [32.3, 35.1, 31.5, 38.8, 33.4, 36.6, 32.2, 34.4, 37.0, 33.3],
    latencyBefore: [0.034, 0.041, 0.029, 0.048, 0.031, 0.038, 0.035, 0.042, 0.030, 0.036],
    jitter: [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.04, 0.05, 0.06, 0.05],
    remedyText: "Increase Redis instance limit or adjust eviction policy to volatile-lru."
  },
  custom: {
    name: "User-Uploaded Custom Logs (custom_incident_logs.json)",
    incidents: 30,
    patternsLearned: 5,
    format: "JSON log entries (stdout / stderr)",
    score: 58,
    successRate: 48.0,
    pai: 58.6,
    overallRisk: "HIGH",
    flags: [
      { level: "critical", metric: "User Override rate", observed: "18.3%", threshold: "<5%", msg: "Unknown error stack pattern. Operator immediately overridden system restart." },
      { level: "critical", metric: "False-Positives", observed: "14.5%", threshold: "<2%", msg: "System restarted multiple core pods due to false latency triggers." },
      { level: "warning", metric: "Low-Confidence Escalations", observed: "12.2%", threshold: "<5%", msg: "Extreme noise in network interface latency logs." }
    ],
    outliers: [3, 11, 24, 38, 45, 49],
    cpuBefore: [85.4, 91.1, 82.8, 89.4, 87.9, 95.0, 81.7, 92.3, 86.5, 88.1],
    latencyBefore: [0.183, 0.201, 0.165, 0.224, 0.179, 0.242, 0.155, 0.211, 0.185, 0.198],
    jitter: [0.22, 0.25, 0.19, 0.28, 0.21, 0.29, 0.18, 0.24, 0.22, 0.23],
    remedyText: "Filter out system noises from error stacks and map clear action thresholds."
  }
};

const CLI_COMMANDS = {
  init: {
    cmd: "mergen init --shadow-mode",
    desc: "Initializes the Mergen configuration and hooks into workspace telemetry in non-intrusive shadow mode.",
    output: `Initializing Mergen Agent v1.2.0 ...
Creating local config file: .env
Checking local environment: OK
Connecting to Kubernetes cluster contexts ...
Discovered 3 operational namespaces: ['production', 'staging', 'infra']
Compiling workspace MCP configurations ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MERGEN INITIALIZATION SUCCESS
  Shadow Mode: Enabled (30-day window)
  Config File: .env
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  },
  patterns: {
    cmd: "mergen patterns compile",
    desc: "Analyzes runbooks, logs, and historical CLI commands to build baseline operational patterns.",
    output: `Scanning terminal command history ...
Parsing 12 local SRE incident runbooks ...
Connecting to Prometheus metrics API ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MERGEN OPERATIONS RECONSTRUCTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Runbooks parsed      : 12
  Commands indexed     : 4,500
  Patterns extracted   : 147
  
  Discovered Action Mappings:
  • Kubernetes Pod crash → auto-restart-policy (Confidence: 94.2%)
  • Connection pool lock → connection-recycle-policy (Confidence: 89.1%)`
  },
  actions: {
    cmd: "mergen actions log --limit 10",
    desc: "Displays proposed and executed actions by the agent under shadow mode, detailing its reasoning logs.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MERGEN SHADOW ACTION REASONING LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Incident: Pod crashloopbackoff in namespace: production
  Proposed Action: Restart pod and inject CPU allocation patch
  Confidence: 94.2%
  Status: BLOCKED (Shadow Mode Only)
  Reasoning: Pod restarted 3 times in 10 mins. Memory footprint stable.
             CPU throttled (Limit: 500m). Increasing limits is safe.

  Incident: DB connections spikes to 92% limit
  Proposed Action: Recycle stale backend processes
  Confidence: 72.4%
  Status: BLOCKED (Confidence threshold 80.0% not met)
  Reasoning: 15 inactive connections found idle for > 20 mins.
             Risk of disconnecting running batch process is high.`
  },
  override: {
    cmd: "mergen override --incident-id 412 --type false-positive",
    desc: "Registers manual feedback to calibrate and adjust the agent's confidence threshold model.",
    output: `Registering manual override feedback ...
Incident ID: 412
Feedback type: FALSE_POSITIVE
Calibrating pattern database ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MERGEN MODEL RE-CALIBRATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Incident Action suppressed: pg-pool-recycle-on-backup
  New Confidence weight: -14.2% (New Confidence: 58.2%)
  Action suppression active during DB backup events.`
  },
  export: {
    cmd: "mergen compliance export --out audit_log.json",
    desc: "Generates an audited compliance report of all agent actions, overrides, and suppressed triggers.",
    output: `Exporting compliance data ...
Writing JSON output to: audit_log.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXPORT COMPLETED
  File: audit_log.json
  Total decisions logged: 312
  Total manual overrides: 5
  Blocked by confidence gate: 23
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [cliTab, setCliTab] = useState('init');
  
  // Simulator states
  const [selectedIncidentKey, setSelectedIncidentKey] = useState('db_pool');
  const [confidenceGate, setConfidenceGate] = useState(0.80);
  const [remedies, setRemedies] = useState({ smooth: true, interpolate: true, trim: false });
  const [simulatedData, setSimulatedData] = useState(INCIDENT_DOMAINS.db_pool);
  const [isUploading, setIsUploading] = useState(false);
  const [calibrationFeedback, setCalibrationFeedback] = useState(null);

  useEffect(() => {
    setSimulatedData(INCIDENT_DOMAINS[selectedIncidentKey]);
    setCalibrationFeedback(null);
  }, [selectedIncidentKey]);

  const handleFileUpload = (e) => {
    e.preventDefault();
    setIsUploading(true);
    setTimeout(() => {
      setSelectedIncidentKey('custom');
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

  const handleCalibration = (isCorrect) => {
    if (isCorrect) {
      setCalibrationFeedback({
        type: 'success',
        msg: 'Decision marked as correct. Positive reinforcement added to pattern weights.'
      });
    } else {
      setCalibrationFeedback({
        type: 'override',
        msg: 'Decision marked as incorrect / false-positive. Pattern suppressed for future backup cycles.'
      });
    }
  };

  // Chart configs
  const chartData = {
    labels: Array.from({ length: 10 }, (_, i) => `Incident ${i + 1}`),
    datasets: [
      {
        label: 'Mean Latency Before (seconds)',
        data: simulatedData.latencyBefore,
        backgroundColor: 'rgba(99, 102, 241, 0.2)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 2,
        borderRadius: 4,
        type: 'bar'
      },
      {
        label: 'Peak CPU Load %',
        data: simulatedData.cpuBefore,
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
        title: { display: true, text: 'Acquisition Latency (s)', color: '#9ca3af' }
      },
      y1: {
        position: 'right',
        grid: { display: false },
        ticks: { color: '#6b7280', callback: (val) => `${(val * 1).toFixed(0)}%` },
        title: { display: true, text: 'CPU Utilization %', color: '#9ca3af' }
      }
    }
  };

  // Simulated Curing Data Before vs After Remediation
  const beforeCureData = [2.4, 8.5, 1.2, 0.8, 14.1, 2.0, 1.1, 7.3, 0.9, 1.4];
  const afterCureData = [1.2, 1.8, 0.9, 0.6, 1.5, 1.1, 0.8, 1.4, 0.7, 1.0];

  const cureChartData = {
    labels: Array.from({ length: 10 }, (_, i) => `Step ${i * 10}`),
    datasets: [
      {
        label: 'Without Mergen Action (Extended Outage)',
        data: beforeCureData,
        borderColor: 'rgba(239, 68, 68, 0.8)',
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.1,
      },
      {
        label: 'With Mergen Auto-Remediation (Rapid Recovery)',
        data: afterCureData,
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
            }}>MERGEN</div>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>Autonomous SRE Operations Agent</span>
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
              <Cpu size={16} /> Control Panel & Shadow Logs
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
                <Zap size={14} className="text-indigo-400" /> Continuous Local Pattern Learning & Telemetry Gating
              </div>

              <h1 style={{ fontSize: '64px', fontWeight: 800, maxWidth: '800px', lineHeight: '1.1', background: 'linear-gradient(to right, #ffffff, #9ca3af)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Autonomous Incident Remediation <br />Built for SRE Teams
              </h1>

              <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '640px', margin: '0 auto' }}>
                Mergen compiles your team's operational patterns from command histories, logs, and runbooks to execute safe, self-calibrating interventions.
              </p>

              <div style={{ display: 'flex', gap: '16px', marginTop: '10px' }}>
                <button onClick={() => setActiveTab('simulator')} className="btn-primary">
                  <Play size={16} /> Open Control Panel
                </button>
                <a href="#setup" className="btn-secondary" style={{ textDecoration: 'none' }} onClick={(e) => { e.preventDefault(); setCliTab('init'); document.getElementById('terminal-section')?.scrollIntoView({ behavior: 'smooth' }); }}>
                  Get Started
                </a>
              </div>
            </section>

            {/* Core Pillars */}
            <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--success)' }}>
                  <ShieldCheck size={24} />
                </div>
                <h3>Grounded Pattern Learning</h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                  Mergen hooks directly into your team's console history and local runbook indexes. Instead of executing generic LLM guesses, it maps specific infrastructure events to known commands to reconstruct safe operational templates.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--success)', fontWeight: 600 }}>
                  <CheckCircle2 size={14} /> Builds tailored patterns locally inside your terminal.
                </div>
              </div>

              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(139, 92, 246, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--brand-violet)' }}>
                  <Cpu size={24} />
                </div>
                <h3>Confidence Gating & Suppression</h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                  Actions are blocked by a custom telemetry gate unless confidence thresholds are exceeded. Outlier detection engines evaluate real-time signals against normal baselines, ensuring backup scripts or scheduled maintenance never trigger false auto-restarts.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--brand-violet)', fontWeight: 600 }}>
                  <Zap size={14} /> Symmetric feedback ensures constant calibration of active gates.
                </div>
              </div>
            </section>

            {/* CLI Commands Tabs interactive */}
            <section id="terminal-section" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div style={{ textAlign: 'left' }}>
                <h2>SRE-First CLI Integration</h2>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Lightweight, command-line interface integrating into SRE terminal contexts via MCP tools.</p>
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
                    <span className="terminal-title">mergen-cli --zsh</span>
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
            <div className="glass-card" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center', justifyContent: 'space-between', padding: '20px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600, uppercase: 'true' }}>Target Pattern Domain</label>
                <select 
                  value={selectedIncidentKey} 
                  onChange={(e) => setSelectedIncidentKey(e.target.value)}
                  className="glass-input"
                  style={{ minWidth: '280px', background: '#0b0f19' }}
                >
                  <option value="db_pool">Postgres Pool Exhaustion (Active Logs)</option>
                  <option value="k8s_crash">Kubernetes Pod CrashLoopBackOff (Active Logs)</option>
                  <option value="redis_evict">Redis Memory Eviction Storm (Active Logs)</option>
                  <option value="custom">Custom Incident Log (User Uploaded)</option>
                </select>
              </div>

              {/* Mock File Uploader */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>or analyze custom events:</span>
                <form onSubmit={handleFileUpload}>
                  <button 
                    disabled={isUploading} 
                    className="btn-secondary" 
                    style={{ fontSize: '13px', padding: '10px 16px', display: 'flex', alignItems: 'center', gap: '8px' }}
                  >
                    <Upload size={14} /> 
                    {isUploading ? "Analyzing..." : "Load Incident Logs"}
                  </button>
                </form>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>INCIDENTS INDEXED</div>
                  <div style={{ fontSize: '16px', fontWeight: 700 }}>{simulatedData.incidents}</div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>PATTERNS COMPILED</div>
                  <div style={{ fontSize: '16px', fontWeight: 700 }}>{simulatedData.patternsLearned}</div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>TELEMETRY SOURCES</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--brand-indigo)' }}>{simulatedData.format}</div>
                </div>
              </div>
            </div>

            {/* Grid Layout Dashboard */}
            <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '30px', alignItems: 'start' }}>
              
              {/* Left Sidebar - Score Gauge & Warnings */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                
                {/* Score panel */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                  <h3 style={{ fontSize: '15px', color: 'var(--text-secondary)' }}>Agent Pattern Accuracy</h3>
                  {renderGauge(simulatedData.score)}
                  
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>AUTONOMOUS RESOLUTION RATE</div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: getScoreColor(simulatedData.score) }}>
                      {simulatedData.successRate}%
                    </div>
                  </div>
                </div>

                {/* Risk Warning List */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                  <h3 style={{ fontSize: '15px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <AlertTriangle size={16} className="text-yellow-500" /> Incident Anomalies & Flags
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
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span className={`status-badge ${flag.level === 'critical' ? 'status-badge-danger' : 'status-badge-warning'}`}>
                            {flag.level.toUpperCase()}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>{flag.metric}</span>
                        </div>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{flag.msg}</p>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          Observed: <strong style={{ color: '#fff' }}>{flag.observed}</strong> (Threshold: {flag.threshold})
                        </div>
                      </div>
                    ))}
                    {simulatedData.flags.length === 0 && (
                      <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '10px' }}>
                        No anomalies logged in current cycle.
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* Right Side - Charts & Tools Panels */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                
                {/* Visual Distribution Chart */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                  <h3>Per-Incident SRE Telemetry Profiler</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Acquisition Latency bounds and CPU utilization spikes parsed across incident logs.</p>
                  
                  <div style={{ height: '280px', width: '100%', position: 'relative' }}>
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                </div>

                {/* Symmetric Action Log / Verdict Calibration & Gate Controls */}
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '30px' }}>
                  
                  {/* Symmetric Feedback / Verdict block */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <ShieldCheck size={18} className="text-success" /> Decision Calibration Verdict
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Reinforce or suppress shadow decisions to calibrate the active confidence engine.
                    </p>

                    <div style={{ display: 'flex', gap: '12px', margin: '5px 0' }}>
                      <button 
                        onClick={() => handleCalibration(true)} 
                        className="btn-secondary" 
                        style={{ border: '1px solid rgba(16, 185, 129, 0.3)', flex: 1, padding: '10px', fontSize: '12px', justifyContent: 'center' }}
                      >
                        Action Correct
                      </button>
                      <button 
                        onClick={() => handleCalibration(false)} 
                        className="btn-secondary" 
                        style={{ border: '1px solid rgba(239, 68, 68, 0.3)', flex: 1, padding: '10px', fontSize: '12px', justifyContent: 'center' }}
                      >
                        Action Wrong / Override
                      </button>
                    </div>

                    {calibrationFeedback && (
                      <div style={{ 
                        fontSize: '12px', 
                        padding: '10px', 
                        borderRadius: '8px', 
                        background: calibrationFeedback.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        border: calibrationFeedback.type === 'success' ? '1px solid rgba(16, 185, 129, 0.2)' : '1px solid rgba(239, 68, 68, 0.2)',
                        color: '#fff'
                      }}>
                        {calibrationFeedback.msg}
                      </div>
                    )}
                  </div>

                  {/* Suppression Gate Panel */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sliders size={18} className="text-violet-400" /> Suppression Gate Threshold
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Set the minimum confidence required to auto-execute resolution policies.
                    </p>

                    <div style={{ display: 'flex', columnGap: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '13px', fontWeight: 500 }}>Min Confidence Gate:</span>
                        <span style={{ color: 'var(--brand-violet)', fontWeight: 700, fontSize: '15px' }}>
                          {(confidenceGate * 100).toFixed(0)}%
                        </span>
                      </div>
                      
                      <input 
                        type="range" 
                        min="0.5" 
                        max="0.95" 
                        step="0.05" 
                        value={confidenceGate} 
                        onChange={(e) => setConfidenceGate(parseFloat(e.target.value))}
                        style={{ accentColor: 'var(--brand-violet)', cursor: 'pointer', width: '100%' }}
                      />

                      <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                        <div>Escalations Suppressed: <strong style={{ color: 'var(--success)' }}>{simulatedData.outliers.length} events</strong></div>
                        <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                        <div>Confidence PAI: <strong style={{ color: 'var(--brand-indigo)' }}>{simulatedData.pai}%</strong></div>
                      </div>
                    </div>
                  </div>

                </div>

                {/* Trajectory Remediation (Curing) Simulator */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px', textAlign: 'left' }}>
                  <div style={{ display: 'flex', justify: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h3>Active Mitigation & Mitigation Policy Simulation</h3>
                      <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Preview latency recovery patterns under automated container orchestration patches.</p>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: '24px' }}>
                    {/* Checkbox configs */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, uppercase: 'true' }}>Remediation Rules</div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.smooth} 
                          onChange={(e) => setRemedies({...remedies, smooth: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Auto-Restart Pods</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.interpolate} 
                          onChange={(e) => setRemedies({...remedies, interpolate: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Scale Replica Allocations</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                        <input 
                          type="checkbox" 
                          checked={remedies.trim} 
                          onChange={(e) => setRemedies({...remedies, trim: e.target.checked})}
                          style={{ accentColor: 'var(--success)' }} 
                        />
                        <span>Recycle Stale Connections</span>
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

                {/* Infrastructure Warnings (Technical Fit Check) */}
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  border: '1px solid rgba(245, 158, 11, 0.2)', 
                  background: 'rgba(245, 158, 11, 0.05)',
                  padding: '12px 16px', 
                  borderRadius: '8px',
                  color: 'var(--text-secondary)',
                  fontSize: '12px',
                  textAlign: 'left'
                }}>
                  <AlertTriangle size={18} className="text-warning" style={{ flexShrink: 0 }} />
                  <div>
                    <strong>Local Environment Status:</strong> Thread-Safe local file lock active (JSON concurrency resolved). To host on a shared server, configure postgres storage backend and enable auth token middleware.
                  </div>
                </div>

              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer Banner */}
      <footer style={{ borderTop: '1px solid var(--border-color)', position: 'absolute', bottom: 0, left: 0, right: 0, height: '60px', display: 'flex', alignItems: 'center', background: 'rgba(3,7,18,0.4)', backdropFilter: 'blur(6px)' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
          <span>© 2026 Mergen Autonomous Operations. Released under the Apache 2.0 License.</span>
          <span>Observability & Auto-Remediation for Physical AI & SRE Workspaces</span>
        </div>
      </footer>
    </div>
  );
}
