import { useState } from 'react'
import {
  ArrowDown,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  Clipboard,
  ExternalLink,
  Github,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import './App.css'

const LINKS = {
  github: 'https://github.com/omertt27/Calibra',
  demo: 'https://huggingface.co/spaces/omert27/robot-dataset-health-check',
  docs: 'https://omertt27.github.io/Calibra/docs/',
  benchmarks: 'https://github.com/omertt27/Calibra#benchmark-results',
  license: 'https://github.com/omertt27/Calibra/blob/main/LICENSE',
  pypi: 'https://pypi.org/project/calibra-robotics/',
}

const BADGES = [
  {
    href: LINKS.github,
    alt: 'GitHub stars',
    src: 'https://img.shields.io/github/stars/omertt27/Calibra?style=flat-square&label=stars&labelColor=17211b&color=bde75a',
  },
  {
    href: LINKS.pypi,
    alt: 'PyPI downloads',
    src: 'https://img.shields.io/pypi/dm/calibra-robotics?style=flat-square&label=downloads%2Fmo&labelColor=17211b&color=bde75a',
  },
]

const INSTALL_COMMAND = 'pip install calibra-robotics'

const STATS = [
  { number: '75%', label: 'less training data', note: 'at matched policy performance' },
  { number: '0.5%', label: 'difference from full-data performance', note: 'LeRobot PushT · 25% retention · 5 seeds' },
  { number: '67%', label: 'better rare-behavior preservation', note: 'vs. random coreset selection' },
]

const WORKFLOW = [
  { number: '01', name: 'Integrity', question: 'Can I trust this dataset?', command: 'calibra integrity' },
  { number: '02', name: 'Quality', question: 'Is it clean?', command: 'calibra audit' },
  { number: '03', name: 'Coverage', question: 'Is it diverse?', command: 'calibra review' },
  { number: '04', name: 'Optimization', question: 'Can I train faster?', command: 'calibra prune' },
]

const ECOSYSTEMS = [
  { name: 'LeRobot', detail: 'v1 / v2 / v3', mark: 'LR' },
  { name: 'Isaac Lab', detail: 'HDF5', logo: 'nvidia.svg' },
  { name: 'RLDS', detail: 'TF Datasets', logo: 'tensorflow.svg' },
  { name: 'robomimic', detail: 'HDF5', mark: 'RM' },
  { name: 'Hugging Face', detail: 'Hub IDs', logo: 'huggingface.svg' },
]

function Logo() {
  return (
    <a className="brand" href="#top" aria-label="Calibra home">
      <img src={`${import.meta.env.BASE_URL}logo-icon.svg`} alt="" />
      <span>Calibra</span>
    </a>
  )
}

function CopyCommand({ large = false }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    await navigator.clipboard.writeText(INSTALL_COMMAND)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <button className={`copy-command${large ? ' copy-command-large' : ''}`} onClick={copy} type="button">
      <code><span>$</span> {INSTALL_COMMAND}</code>
      <span className="copy-action">
        {copied ? <Check size={16} /> : <Clipboard size={16} />}
        {copied ? 'Copied' : 'Copy'}
      </span>
    </button>
  )
}

function TerminalReport() {
  return (
    <div className="terminal" aria-label="Real calibra integrity CLI output">
      <div className="terminal-bar">
        <div className="terminal-dots"><i /><i /><i /></div>
        <span>calibra — integrity</span>
        <span className="terminal-status">completed in 0.8s</span>
      </div>
      <div className="terminal-body">
        <div className="terminal-command"><span>$</span> calibra integrity pusht_demo</div>
        <p className="report-meta">pusht_demo · 20 episodes</p>

        <div className="report-group report-critical">
          <div className="report-label"><ShieldAlert size={15} /> Critical <span>1</span></div>
          <p><span>short_episode_fraction</span><strong>4 / 20 episodes</strong></p>
          <p className="report-detail">20.0% of episodes have fewer than 80 steps (lower IQR fence) — likely aborted demonstrations. BC policies trained on these learn to quit tasks early.</p>
        </div>

        <div className="report-group report-passed">
          <div className="report-label"><CheckCircle2 size={15} /> Passed <span>6</span></div>
          <p><span>timestamp_jitter_cv</span><strong>pass</strong></p>
          <p><span>action_dropout_rate</span><strong>pass</strong></p>
          <p><span>ldlj</span><strong>pass</strong></p>
          <p><span>jerk_spike_rate</span><strong>pass</strong></p>
        </div>

        <div className="report-group report-skipped">
          <div className="report-label"><Circle size={15} /> Not evaluated <span>3</span></div>
          <p><span>camera_freeze</span><strong>requires images</strong></p>
          <p><span>duplicate_frame</span><strong>requires images</strong></p>
        </div>

        <div className="report-footer">
          <span>Integrity Score: <strong>86 / 100</strong> · Status: Warning</span>
          <span className="report-ci">CI result: Failed — 1 critical finding</span>
        </div>
      </div>
    </div>
  )
}

function StatsBar() {
  return (
    <div className="stats-bar">
      <div className="container stats-grid">
        {STATS.map((stat) => (
          <div className="stat-item" key={stat.number}>
            <strong className="stat-number">{stat.number}</strong>
            <span className="stat-label">{stat.label}</span>
            <span className="stat-note">{stat.note}</span>
          </div>
        ))}
      </div>
      <div className="container">
        <p className="stats-caveat">
          Measured on LeRobot PushT. Curation gains hold up across a broader 5-seed sweep on
          3 datasets and 3 policy families (BC-MLP, ACT, Diffusion Policy) —{' '}
          <a href={LINKS.benchmarks} target="_blank" rel="noreferrer">see the full benchmark table</a>.
        </p>
      </div>
    </div>
  )
}

const TAIL_COVERAGE = [
  { label: 'Calibra coreset', value: 56.0, className: 'highlight' },
  { label: 'Random baseline', value: 33.6, className: '' },
]

function CoverageGraphic() {
  return (
    <div className="coverage-card">
      <div className="coverage-head">
        <div><span>Tail-behavior coverage</span><strong>41/165 episodes · LeRobot PushT</strong></div>
        <span className="coverage-score">+67% vs. random</span>
      </div>
      <div className="coverage-bars">
        {TAIL_COVERAGE.map((row) => (
          <div className={`coverage-bar-row ${row.className}`} key={row.label}>
            <span>{row.label}</span>
            <div><i style={{ width: `${row.value}%` }} /></div>
            <strong>{row.value.toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <p className="coverage-footnote">
        Share of held-out rare-behavior episodes represented in a 41-episode coreset at 25%
        retention — same benchmark run shown above.
      </p>
    </div>
  )
}

function App() {
  return (
    <div id="top">
      <nav className="nav">
        <div className="container nav-inner">
          <Logo />
          <div className="nav-links">
            <a href="#benchmark">Benchmark</a>
            <a href="#workflow">How it works</a>
            <a href={LINKS.docs}>Docs</a>
          </div>
          <div className="nav-right">
            <div className="nav-badges">
              {BADGES.map((badge) => (
                <a href={badge.href} key={badge.alt} target="_blank" rel="noreferrer">
                  <img src={badge.src} alt={badge.alt} height={20} />
                </a>
              ))}
            </div>
            <a className="nav-github" href={LINKS.github} target="_blank" rel="noreferrer">
              <Github size={17} /> GitHub
            </a>
          </div>
        </div>
      </nav>

      <main>
        <header className="hero">
          <div className="container hero-grid">
            <div className="hero-copy">
              <div className="eyebrow"><span /> Open-source · CI-ready · No account required</div>
              <h1>Stop wasting GPU hours on robot data.</h1>
              <p>
                Calibra audits dataset integrity, measures quality and coverage, and builds
                quality-aware coresets — so you train on less data, spend less compute, and
                know exactly why before training begins.
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                  Try the demo <ExternalLink size={16} />
                </a>
              </div>
              <div className="hero-meta-row">
                <CopyCommand />
                <a className="hero-github-link" href={LINKS.github} target="_blank" rel="noreferrer">
                  <Github size={16} /> View on GitHub
                </a>
              </div>
            </div>
            <TerminalReport />
          </div>
          <a className="scroll-cue" href="#benchmark">
            See benchmark results <ArrowDown size={15} />
          </a>
        </header>

        <StatsBar />

        <section className="research section" id="benchmark">
          <div className="container">
            <div className="research-card">
              <div className="research-copy">
                <div className="research-mark"><Sparkles size={22} /></div>
                <div className="eyebrow"><span /> Benchmark · LeRobot PushT</div>
                <h2><strong>75% smaller</strong> dataset. Prediction error within 0.5% of full-data training.</h2>
                <p>
                  Calibra retained 41 of 165 training episodes while preserving more
                  action-space tail coverage than a random coreset — and matching the
                  full-dataset baseline across 5 seeds.
                </p>
                <div className="benchmark-meta">
                  <span>5 seeds</span><span>120 epochs</span><span>BC-MLP</span><span>25% retained</span>
                </div>
                <a className="text-link" href={LINKS.benchmarks} target="_blank" rel="noreferrer">
                  Full benchmarks and limitations <ArrowRight size={16} />
                </a>
              </div>
              <div className="benchmark-card">
                <div className="benchmark-title">
                  <span>LeRobot PushT · test MSE</span>
                  <small>Lower is better</small>
                </div>
                <div className="benchmark-row">
                  <span>Full dataset <small>165 episodes</small></span>
                  <div><i style={{ width: '100%' }} /></div>
                  <strong>420.93</strong>
                </div>
                <div className="benchmark-row highlight">
                  <span>Calibra <small>41 episodes</small></span>
                  <div><i style={{ width: '99.6%' }} /></div>
                  <strong>422.77</strong>
                </div>
                <div className="benchmark-row">
                  <span>Random <small>41 episodes</small></span>
                  <div><i style={{ width: '97.9%' }} /></div>
                  <strong>429.61</strong>
                </div>
                <div className="tail-coverage">
                  <div><span>Calibra tail coverage</span><strong>56.0%</strong></div>
                  <div><span>Random tail coverage</span><strong>33.6%</strong></div>
                </div>
                <p>Source: 5-seed retention sweep. Full-data MSE is the unfiltered 165-episode baseline.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="workflow section" id="workflow">
          <div className="container">
            <div className="section-intro">
              <div className="eyebrow"><span /> One pipeline, in the right order</div>
              <h2>Fix the data question before the model question.</h2>
              <p>Calibra turns an unknown dataset into a training decision through four explicit checks.</p>
            </div>
            <div className="workflow-grid">
              {WORKFLOW.map((step, index) => (
                <article className="workflow-step" key={step.name}>
                  <div className="step-top">
                    <span className="step-number">{step.number}</span>
                    {index < WORKFLOW.length - 1 && <ArrowRight className="step-arrow" size={20} />}
                  </div>
                  <h3>{step.name}</h3>
                  <p>{step.question}</p>
                  <code>{step.command}</code>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="proof section">
          <div className="container proof-grid">
            <div className="proof-copy">
              <div className="eyebrow"><span /> Evidence, not a black box</div>
              <h2>See the problem. Find the episode. Fix it.</h2>
              <p>
                Every warning maps to a measurable condition and the exact episodes that triggered it.
                Calibra runs locally and produces deterministic results you can inspect.
              </p>
              <ul>
                <li><CheckCircle2 size={18} /> Episode-level root causes</li>
                <li><CheckCircle2 size={18} /> CI-friendly exit codes</li>
                <li><CheckCircle2 size={18} /> No upload, account, or API key</li>
              </ul>
              <a className="text-link" href={LINKS.docs}>Explore the commands <ArrowRight size={16} /></a>
            </div>
            <CoverageGraphic />
          </div>
        </section>

        <section className="ecosystems section">
          <div className="container">
            <div className="section-intro compact">
              <div className="eyebrow"><span /> Works with your stack</div>
              <h2>Start with the data you already have.</h2>
            </div>
            <div className="ecosystem-list">
              {ECOSYSTEMS.map((ecosystem) => (
                <div className="ecosystem" key={ecosystem.name}>
                  <div className="ecosystem-logo">
                    {ecosystem.logo
                      ? <img src={`${import.meta.env.BASE_URL}${ecosystem.logo}`} alt="" loading="lazy" />
                      : <span>{ecosystem.mark}</span>}
                  </div>
                  <div>
                    <strong>{ecosystem.name}</strong>
                    <span>{ecosystem.detail}</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="support-note">
              Camera-frame integrity checks currently support HDF5 datasets and LeRobot v1 with image decoding.
            </p>
          </div>
        </section>

        <section className="demo section" id="demo">
          <div className="container demo-grid">
            <div className="demo-copy">
              <div className="eyebrow"><span /> Try it in the browser</div>
              <h2>Inspect a public LeRobot dataset without installing anything.</h2>
              <p>
                Enter a Hub dataset ID to see integrity checks, quality findings, community
                comparisons, and a recommended keep fraction.
              </p>
              <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                Open the live demo <ExternalLink size={16} />
              </a>
            </div>
            <a className="demo-frame" href={LINKS.demo} target="_blank" rel="noreferrer">
              <div className="demo-frame-bar">
                <span><i /><i /><i /></span>
                huggingface.co/spaces/omert27/robot-dataset-health-check
              </div>
              <img src={`${import.meta.env.BASE_URL}hf-space.png`} alt="Calibra Robot Dataset Health Check on Hugging Face Spaces" />
            </a>
          </div>
        </section>

        <section className="final-cta section">
          <div className="container">
            <div className="cta-panel">
              <span className="cta-kicker">Before your next training run</span>
              <h2>Check the data first.</h2>
              <p>Install Calibra locally or inspect a public LeRobot dataset in the browser — free, no account needed.</p>
              <CopyCommand large />
              <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                Try the Hugging Face demo <ExternalLink size={16} />
              </a>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="container footer-inner">
          <Logo />
          <p>Train on less data. Spend less compute. Ship better policies.</p>
          <div>
            <a href={LINKS.github}>GitHub</a>
            <a href={LINKS.docs}>Docs</a>
            <a href={LINKS.license}>BSL 1.1</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
