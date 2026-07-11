import React, { useState, useEffect, useRef } from 'react';
import {
  Github, BookOpen, FileText, BarChart3, Terminal, ShieldCheck, Layers,
  Scissors, Activity, GitCompare, Copy, Check, ArrowRight, Zap,
  Database, FlaskConical, ScrollText, GaugeCircle, Menu, X, Cpu,
} from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Tooltip, Legend, Filler,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

/* ------------------------------------------------------------------ */
/*  Constants — single source of truth for outbound links.            */
/* ------------------------------------------------------------------ */
const LINKS = {
  github: 'https://github.com/omerTT/Calibra',
  // arXiv preprint pending — points at the LaTeX source until the ID lands.
  paper: 'https://github.com/omerTT/Calibra/tree/main/paper',
  docs: 'https://omerTT.github.io/Calibra/',
  pypi: 'https://pypi.org/project/calibra-robotics/',
  contact: 'mailto:omertahtoko@gmail.com',
};
const INSTALL_CMD = 'pip install calibra-robotics';

/* ------------------------------------------------------------------ */
/*  Data — every number below is drawn verbatim from README.md.       */
/* ------------------------------------------------------------------ */
const STATS = [
  { num: '3', unit: '×', label: 'policy architectures validated (BC-MLP, ACT, Diffusion)' },
  { num: '16', unit: '', label: 'reference datasets shipped' },
  { num: '41.7', unit: '%', label: 'better than full-data at 10% keep (PushT real)' },
  { num: '596', unit: '', label: 'tests, deterministic — not a language model' },
];

const COMMANDS = [
  { name: 'calibra audit', icon: BarChart3, desc: 'Full diagnostic report — four analyzers, bootstrap CIs, per-episode outlier detection.' },
  { name: 'calibra compare', icon: GitCompare, desc: 'Evidence-backed comparison against a reference dataset. Every claim is falsifiable.' },
  { name: 'calibra certify', icon: ShieldCheck, desc: 'Pass / provisional / fail quality gate with CI exit codes and a remediation checklist.' },
  { name: 'calibra prune', icon: Scissors, desc: 'Two-stage coreset selection: quality filter + greedy max-coverage over ~50k episodes.' },
  { name: 'calibra predict', icon: GaugeCircle, desc: 'Predict downstream training success before spending a single GPU-hour.' },
  { name: 'calibra watch', icon: Activity, desc: 'Real-time teleoperation monitor — flags a bad episode seconds after it is saved.' },
];

// Retention curves — "% improvement over random selection" at each keep fraction.
const KEEPS = ['10%', '20%', '30%', '50%', '70%'];
const RETENTION = {
  pusht: [56.6, 49.1, 42.5, 32.2, 15.0],
  aloha: [50.0, 35.5, 29.2, 7.8, 7.2],
  droid: [20.9, 15.2, 10.1, 7.2, 4.5],
};

// Cross-architecture mean improvement over random (keep 30%, 5 seeds, 3 datasets).
const ARCH_TABLE = [
  { method: 'Diversity-only', bc: '+29.5%', act: '+26.5%', diff: '+11.9%' },
  { method: 'Calibra full', bc: '+24.5%', act: '+23.7%', diff: '+13.8%', hl: true },
  { method: 'K-Center greedy', bc: '+24.0%', act: '+23.1%', diff: '+10.1%' },
  { method: 'Facility Location', bc: '+21.5%', act: '+18.4%', diff: '+8.7%' },
  { method: 'Quality-filter only', bc: '+16.2%', act: '+17.4%', diff: '+11.1%' },
  { method: 'Random', bc: '0.0%', act: '0.0%', diff: '0.0%' },
  { method: 'Herding', bc: '−11.4%', act: '−7.5%', diff: '−5.5%' },
];

const FORMATS = ['LeRobot v2 (Parquet)', 'LeRobot v1', 'HuggingFace Hub', 'HDF5 · Isaac Lab', 'Robomimic', 'RLDS / TFDS', 'MCAP / ROS2'];

/* ------------------------------------------------------------------ */
/*  Small building blocks                                             */
/* ------------------------------------------------------------------ */
function Logo({ size = 30 }) {
  return (
    <>
      <img src="/logo-icon.svg" alt="" width={size} height={size} />
      <span className="logo-wordmark">Calib<span>ra</span></span>
    </>
  );
}

function InstallPill({ onCopy }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(INSTALL_CMD);
    setCopied(true);
    onCopy?.();
    setTimeout(() => setCopied(false), 1600);
  };
  return (
    <div className="install-pill" onClick={copy} role="button" tabIndex={0}
         onKeyDown={(e) => e.key === 'Enter' && copy()}>
      <span className="dollar">$</span>
      <span>{INSTALL_CMD}</span>
      <span className="copy-btn">{copied ? <Check size={15} /> : <Copy size={15} />}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sections                                                          */
/* ------------------------------------------------------------------ */
function Nav() {
  const [open, setOpen] = useState(false);
  const item = (href, label, ext = true) => (
    <a className="nav-link" href={href} onClick={() => setOpen(false)}
       {...(ext ? { target: '_blank', rel: 'noreferrer' } : {})}>{label}</a>
  );
  return (
    <nav className="nav">
      <div className="container nav-inner">
        <div className="nav-brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
          <Logo />
        </div>
        <div className={`nav-links ${open ? 'open' : ''}`}>
          {item('#benchmarks', 'Benchmarks', false)}
          {item(LINKS.paper, 'Paper')}
          {item(LINKS.docs, 'Docs')}
          {item(LINKS.github, 'GitHub')}
          <a className="btn-primary nav-cta" href={LINKS.github} target="_blank" rel="noreferrer">
            <Github size={16} /> Star on GitHub
          </a>
        </div>
        <button className="nav-toggle" onClick={() => setOpen(!open)} aria-label="Menu">
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>
    </nav>
  );
}

function Hero({ onCopy }) {
  return (
    <header className="hero">
      <div className="container hero-grid">
        <div>
          <span className="eyebrow"><Cpu size={14} /> Dataset observability for robot learning</span>
          <h1>Know what's wrong with your robot data <span className="accent">before you train on it.</span></h1>
          <p className="hero-sub">
            Calibra diagnoses kinematic and temporal defects in imitation-learning
            demonstrations, then prunes the redundant episodes — so you stop burning
            GPU-hours on data that was never going to help.
          </p>
          <div className="hero-actions">
            <InstallPill onCopy={onCopy} />
          </div>
          <div className="hero-actions">
            <a className="btn-primary" href="#benchmarks"><BarChart3 size={17} /> See the benchmarks</a>
            <a className="btn-secondary" href={LINKS.github} target="_blank" rel="noreferrer">
              <Github size={17} /> View source
            </a>
          </div>
          <div className="hero-meta">
            <span><ShieldCheck size={15} /> Runs entirely locally</span>
            <span><Zap size={15} /> Deterministic, not an LLM</span>
            <span><ScrollText size={15} /> BSL 1.1 · source-available</span>
          </div>
        </div>

        <div className="hero-terminal glass-card" style={{ padding: 0 }}>
          <div className="terminal-window" style={{ border: 0, boxShadow: 'none' }}>
            <div className="terminal-header">
              <div className="terminal-dots">
                <span className="terminal-dot terminal-dot-red" />
                <span className="terminal-dot terminal-dot-yellow" />
                <span className="terminal-dot terminal-dot-green" />
              </div>
              <span className="terminal-title">calibra compare</span>
            </div>
            <div className="terminal-body">
              <div><span className="t-gold">$</span> <span className="t-white">calibra compare </span><span className="t-blue">my_dataset aloha</span></div>
              <br />
              <div className="t-rule">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
              <div className="t-dim">Reference: aloha_mobile_cabinet · 14D · 85 eps</div>
              <div className="t-dim">Yours:&nbsp;&nbsp;&nbsp;&nbsp;my_dataset · 120 eps</div>
              <div className="t-rule">────────────────────────────────────</div>
              <div className="t-white">VELOCITY DISCONTINUITY RATE</div>
              <div className="t-dim">&nbsp;&nbsp;Yours: <span className="t-red">12.1%</span>&nbsp;&nbsp;aloha: 1.3%&nbsp;&nbsp;<span className="t-red">▲ +10.8%</span></div>
              <br />
              <div className="t-white">JERK SPIKE RATE</div>
              <div className="t-dim">&nbsp;&nbsp;Yours: <span className="t-red">8.4%</span>&nbsp;&nbsp;&nbsp;aloha: 0.7%&nbsp;&nbsp;<span className="t-red">▲ +7.7%</span></div>
              <div className="t-rule">────────────────────────────────────</div>
              <div className="t-gold">RECOMMENDED ACTIONS</div>
              <div className="t-green">&nbsp;&nbsp;Prune episodes 14, 22, 41 — jerk outliers</div>
              <div className="t-green">&nbsp;&nbsp;(MAD analysis). Investigate command</div>
              <div className="t-green">&nbsp;&nbsp;packet drops or operator corrections.</div>
              <div className="t-rule">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function StatBar() {
  return (
    <div className="container section-tight">
      <div className="stat-bar">
        {STATS.map((s) => (
          <div className="stat-cell" key={s.label}>
            <div className="stat-num">{s.num}<span className="unit">{s.unit}</span></div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

const PROBLEMS = [
  { icon: Database, title: 'Silent bad data', body: 'Jerk spikes, dropped frames, comms lag and stuck actuators all look like valid training signal to your policy. It learns the noise.' },
  { icon: Layers, title: 'Wasted compute', body: 'In a 10,000-episode set, 60–80% of episodes are near-duplicates. GPU cost scales with volume, not with the unique behavior you actually need.' },
  { icon: FlaskConical, title: 'Undiagnosable failure', body: 'When a policy stalls, you cannot tell whether the cause is the architecture, the recipe, or the data. Calibra isolates the data variable.' },
];

function Problem() {
  return (
    <section className="section">
      <div className="container">
        <div className="section-head">
          <span className="eyebrow">The problem</span>
          <h2>Training on all your demonstrations is the expensive mistake</h2>
          <p>More data is not free, and it is not always better. Calibra tells you which episodes are hurting you before the run, not after.</p>
        </div>
        <div className="card-grid">
          {PROBLEMS.map((p) => (
            <div className="glass-card feature-card" key={p.title}>
              <div className="feature-icon"><p.icon size={22} /></div>
              <h3>{p.title}</h3>
              <p>{p.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Commands() {
  return (
    <section className="section" style={{ background: 'var(--bg-secondary)' }}>
      <div className="container">
        <div className="section-head">
          <span className="eyebrow"><Terminal size={14} /> One CLI</span>
          <h2>From raw demonstrations to a training-ready coreset</h2>
          <p>Fourteen commands, all local, all deterministic. The six you will reach for most:</p>
        </div>
        <div className="cmd-grid">
          {COMMANDS.map((c) => (
            <div className="cmd-row" key={c.name}>
              <div className="cmd-name">{c.name}</div>
              <div className="cmd-desc">{c.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Benchmarks() {
  const chartData = {
    labels: KEEPS,
    datasets: [
      mkLine('PushT real', RETENTION.pusht, '#f59e0b'),
      mkLine('ALOHA mobile', RETENTION.aloha, '#38bdf8'),
      mkLine('DROID-100', RETENTION.droid, '#a78bfa'),
    ],
  };
  return (
    <section className="section" id="benchmarks">
      <div className="container">
        <div className="section-head">
          <span className="eyebrow"><BarChart3 size={14} /> Empirical validation</span>
          <h2>Coverage selection holds across architectures — and we say where it doesn't</h2>
          <p>
            Every result is from real GPU training on an RTX 2080, 5 shared seeds, paired
            <em> t</em>-tests. Reported at the strength the evidence supports.
          </p>
        </div>

        <div className="bench-layout">
          <div className="glass-card bench-chart-card">
            <h3>Advantage over random selection</h3>
            <div className="cap">% improvement of a Calibra coreset vs. a random coreset, by keep fraction.</div>
            <div className="chart-holder"><Line data={chartData} options={CHART_OPTS} /></div>
          </div>

          <div>
            <div className="cap" style={{ marginBottom: 12, color: 'var(--text-secondary)', fontSize: 14 }}>
              Mean improvement over random — <strong style={{ color: '#fff' }}>identical coresets, only the learner changes</strong> (keep 30%, 5 seeds, 3 datasets):
            </div>
            <div className="bench-table-wrap">
              <table className="bench">
                <thead>
                  <tr><th>Selection method</th><th>BC-MLP</th><th>ACT</th><th>Diffusion</th></tr>
                </thead>
                <tbody>
                  {ARCH_TABLE.map((r) => (
                    <tr key={r.method} className={r.hl ? 'highlight' : ''}>
                      <td>{r.method}</td>
                      <td className={cls(r.bc)}>{r.bc}</td>
                      <td className={cls(r.act)}>{r.act}</td>
                      <td className={cls(r.diff)}>{r.diff}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 10, fontSize: 12.5, color: 'var(--text-muted)' }}>
              Method-ranking agreement across learners: Spearman ρ ≥ 0.86 (BC↔ACT = 1.00).
            </div>
          </div>
        </div>

        <div className="callout" style={{ marginTop: 40 }}>
          <ScrollText className="ic" size={22} />
          <p>
            <strong>We publish the negatives.</strong> Calibra's diversity stage sometimes
            beats the full pipeline; the quality filter helps under corruption and on the
            diffusion learner but is a slight drag on clean deterministic policies; DROID
            morphology-collapse is an open problem. Single-seed runs are labelled
            exploratory. That honesty is the point — every interpretation ships with a
            falsifiable claim and a stated falsification condition.
          </p>
        </div>
      </div>
    </section>
  );
}

const REPRO = [
  { icon: ScrollText, title: 'Falsifiable claim registry', body: 'Every interpretation is backed by a claim in calibra/claims/ with an evidence count, confidence rating and the exact data that would disprove it.' },
  { icon: FlaskConical, title: 'One-command reproduction', body: 'Each benchmark in the README ships with the exact command, dataset ID, seed count and epoch budget. Run it on your own hardware.' },
  { icon: GaugeCircle, title: 'Corruption self-tests', body: 'calibra corrupt injects known defects into clean data to prove each metric actually responds to the fault it claims to detect.' },
];

function Reproducibility() {
  return (
    <section className="section" style={{ background: 'var(--bg-secondary)' }}>
      <div className="container">
        <div className="section-head">
          <span className="eyebrow"><BookOpen size={14} /> Built for labs</span>
          <h2>The only question that matters: can another lab reproduce it?</h2>
          <p>Calibra is designed so the answer is yes. No cloud, no account, no black box.</p>
        </div>
        <div className="repro-grid">
          {REPRO.map((r) => (
            <div className="glass-card repro-card" key={r.title}>
              <h3><r.icon size={18} className="t-gold" /> {r.title}</h3>
              <p>{r.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Formats() {
  return (
    <section className="section-tight">
      <div className="container">
        <div className="section-head" style={{ marginBottom: 32 }}>
          <span className="eyebrow"><Layers size={14} /> Works with your stack</span>
          <h2 style={{ fontSize: 26 }}>Point it at a Hub ID, a path, or an <code className="bench-mono">hf://</code> URI</h2>
        </div>
        <div className="format-row">
          {FORMATS.map((f) => (
            <span className="format-chip" key={f}><span className="dot" /> {f}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTA({ onCopy }) {
  return (
    <section className="section">
      <div className="container cta-band">
        <div className="cta-card">
          <span className="eyebrow"><Terminal size={14} /> Get started</span>
          <h2>Run it before your next training job</h2>
          <p>Free for research and internal use. One pip install away.</p>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 26 }}>
            <InstallPill onCopy={onCopy} />
          </div>
          <div className="cta-actions">
            <a className="btn-primary" href={LINKS.docs} target="_blank" rel="noreferrer">
              <BookOpen size={17} /> Read the docs <ArrowRight size={15} />
            </a>
            <a className="btn-secondary" href={LINKS.paper} target="_blank" rel="noreferrer">
              <FileText size={17} /> Read the paper
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div className="footer-col footer-about">
            <div className="nav-brand"><Logo size={26} /></div>
            <p>Dataset observability and coreset selection for robotics imitation learning. Source-available, local-first.</p>
          </div>
          <div className="footer-col">
            <h4>Product</h4>
            <a href="#benchmarks">Benchmarks</a>
            <a href={LINKS.docs} target="_blank" rel="noreferrer">Documentation</a>
            <a href={LINKS.pypi} target="_blank" rel="noreferrer">PyPI package</a>
          </div>
          <div className="footer-col">
            <h4>Research</h4>
            <a href={LINKS.paper} target="_blank" rel="noreferrer">Paper (preprint)</a>
            <a href={`${LINKS.github}/blob/main/README.md#empirical-validation`} target="_blank" rel="noreferrer">Results</a>
            <a href={`${LINKS.github}/tree/main/calibra/claims`} target="_blank" rel="noreferrer">Claim registry</a>
          </div>
          <div className="footer-col">
            <h4>Project</h4>
            <a href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>
            <a href={`${LINKS.github}/blob/main/LICENSE`} target="_blank" rel="noreferrer">License (BSL 1.1)</a>
            <a href={LINKS.contact}>Contact</a>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} Calibra · Business Source License 1.1 → Apache 2.0 on 2030-06-30</span>
          <a href={LINKS.contact}>omertahtoko@gmail.com</a>
        </div>
      </div>
    </footer>
  );
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
function mkLine(label, data, color) {
  return {
    label, data,
    borderColor: color,
    backgroundColor: color + '22',
    pointBackgroundColor: color,
    pointRadius: 3,
    pointHoverRadius: 5,
    borderWidth: 2.5,
    tension: 0.3,
    fill: false,
  };
}
const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 12 }, usePointStyle: true, boxWidth: 7 } },
    tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: +${c.parsed.y}% vs random` } },
  },
  scales: {
    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'episodes kept', color: '#64748b', font: { size: 11 } } },
    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', callback: (v) => '+' + v + '%' }, beginAtZero: true },
  },
};
function cls(v) { return v.startsWith('+') ? 'pos' : v.startsWith('−') || v.startsWith('-') ? 'neg' : ''; }

/* ------------------------------------------------------------------ */
/*  Root                                                              */
/* ------------------------------------------------------------------ */
export default function App() {
  const [toast, setToast] = useState(false);
  const timer = useRef(null);
  const showToast = () => {
    setToast(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setToast(false), 1800);
  };
  useEffect(() => () => clearTimeout(timer.current), []);

  return (
    <>
      <div className="grid-overlay" />
      <div className="bg-glow-orb bg-glow-top-left" />
      <div className="bg-glow-orb bg-glow-bottom-right" />

      <Nav />
      <main>
        <Hero onCopy={showToast} />
        <StatBar />
        <hr className="section-divider" />
        <Problem />
        <Commands />
        <Benchmarks />
        <Reproducibility />
        <Formats />
        <CTA onCopy={showToast} />
      </main>
      <Footer />

      {toast && (
        <div className="toast"><Check size={16} className="t-gold" /> Copied <code style={{ fontFamily: 'var(--font-mono)' }}>{INSTALL_CMD}</code></div>
      )}
    </>
  );
}
