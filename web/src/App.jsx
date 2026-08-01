import { useState } from 'react'
import {
  ArrowDown,
  ArrowRight,
  Check,
  CheckCircle2,
  Clipboard,
  ExternalLink,
  Github,
  ShieldAlert,
  Sparkles,
  TriangleAlert,
} from 'lucide-react'
import './App.css'

const LINKS = {
  github: 'https://github.com/omertt27/Calibra',
  demo: 'https://huggingface.co/spaces/omert27/robot-dataset-health-check',
  docs: 'https://omertt27.github.io/Calibra/docs/',
  benchmarks: 'https://github.com/omertt27/Calibra#benchmark-results',
  license: 'https://github.com/omertt27/Calibra/blob/main/LICENSE',
}

const INSTALL_COMMAND = 'pip install calibra-robotics'

const WORKFLOW = [
  { number: '01', name: 'Integrity', question: 'Can I trust this dataset?', command: 'calibra integrity' },
  { number: '02', name: 'Quality', question: 'Is it clean?', command: 'calibra audit' },
  { number: '03', name: 'Coverage', question: 'Is it diverse?', command: 'calibra review' },
  { number: '04', name: 'Optimization', question: 'Can I train faster?', command: 'calibra prune' },
]

const ECOSYSTEMS = [
  { name: 'LeRobot', detail: 'v1 / v2 / v3', logo: 'huggingface.svg' },
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
    <div className="terminal" aria-label="Example Calibra integrity report">
      <div className="terminal-bar">
        <div className="terminal-dots"><i /><i /><i /></div>
        <span>calibra — integrity</span>
        <span className="terminal-status">completed in 2.4s</span>
      </div>
      <div className="terminal-body">
        <div className="terminal-command"><span>$</span> calibra integrity /data/my_demos.h5</div>
        <div className="report-heading">
          <span>Dataset Integrity</span>
          <strong>85 / 100</strong>
        </div>
        <p className="report-meta">my_demos · 120 episodes · Status: Warning</p>

        <div className="report-group report-critical">
          <div className="report-label"><ShieldAlert size={15} /> Critical <span>1</span></div>
          <p><span>camera_freeze_events</span><strong>episode ep_17</strong></p>
          <p className="report-detail">1 of 120 episodes contains a run of ≥5 near-identical frames.</p>
        </div>

        <div className="report-group report-warning">
          <div className="report-label"><TriangleAlert size={15} /> Warnings <span>1</span></div>
          <p><span>blurry_episode_fraction</span><strong>1 episode</strong></p>
        </div>

        <div className="report-group report-passed">
          <div className="report-label"><CheckCircle2 size={15} /> Passed <span>8</span></div>
          <p><span>timestamp_jitter_cv</span><strong>pass</strong></p>
          <p><span>timestamp_dropout_rate</span><strong>pass</strong></p>
          <p><span>action_dropout_rate</span><strong>pass</strong></p>
        </div>
      </div>
    </div>
  )
}

function CoverageGraphic() {
  const clusters = [
    ['12%', '18%'], ['20%', '26%'], ['15%', '33%'], ['25%', '15%'],
    ['72%', '17%'], ['80%', '26%'], ['68%', '32%'], ['84%', '13%'],
    ['14%', '70%'], ['23%', '80%'], ['28%', '67%'], ['10%', '84%'],
    ['69%', '72%'], ['79%', '82%'], ['86%', '68%'], ['73%', '88%'],
  ]
  const selected = [1, 5, 10, 14]

  return (
    <div className="coverage-card">
      <div className="coverage-head">
        <div><span>Behavioral coverage</span><strong>Coreset · 25% retained</strong></div>
        <span className="coverage-score">4 / 4 regions</span>
      </div>
      <div className="coverage-plot" aria-label="Calibra selects demonstrations across all behavioral regions">
        <span className="axis-y">Behavior B</span>
        <span className="axis-x">Behavior A</span>
        <div className="quadrant vertical" />
        <div className="quadrant horizontal" />
        {clusters.map(([left, top], index) => (
          <i
            className={selected.includes(index) ? 'selected' : ''}
            key={`${left}-${top}`}
            style={{ left, top }}
          />
        ))}
      </div>
      <div className="coverage-legend">
        <span><i /> Full dataset</span>
        <span><i className="selected" /> Calibra selection</span>
        <strong>Rare behaviors preserved</strong>
      </div>
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
            <a href="#workflow">How it works</a>
            <a href="#research">Research</a>
            <a href={LINKS.docs}>Docs</a>
          </div>
          <a className="nav-github" href={LINKS.github} target="_blank" rel="noreferrer">
            <Github size={17} /> GitHub
          </a>
        </div>
      </nav>

      <main>
        <header className="hero">
          <div className="hero-glow" />
          <div className="container hero-grid">
            <div className="hero-copy">
              <div className="eyebrow"><span /> Dataset observability for robot learning</div>
              <h1>Know what’s wrong with your robot data <em>before training starts.</em></h1>
              <p>
                A source-available CLI for robotics teams using LeRobot, Isaac Lab, and robomimic.
                Detect dataset problems, understand coverage, and optimize training data before spending GPU time.
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                  Try the demo <ExternalLink size={16} />
                </a>
                <a className="button button-secondary" href={LINKS.github} target="_blank" rel="noreferrer">
                  <Github size={17} /> View on GitHub
                </a>
              </div>
              <CopyCommand />
            </div>
            <TerminalReport />
          </div>
          <a className="scroll-cue" href="#workflow">
            See how it works <ArrowDown size={15} />
          </a>
        </header>

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

        <section className="research section" id="research">
          <div className="container">
            <div className="research-card">
              <div className="research-copy">
                <div className="research-mark"><Sparkles size={22} /></div>
                <div className="eyebrow"><span /> Research</div>
                <h2><strong>75% smaller</strong> with prediction error within 0.5% of full-data training.</h2>
                <p>
                  On LeRobot PushT, Calibra retained 41 of 165 training episodes while preserving
                  more action-space tail coverage than a random coreset.
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

        <section className="final-cta section">
          <div className="container">
            <div className="cta-panel">
              <span className="cta-kicker">Before your next training run</span>
              <h2>Check the data first.</h2>
              <p>Install Calibra locally or inspect a public LeRobot dataset in the browser.</p>
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
          <p>Dataset observability for robot learning.</p>
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
