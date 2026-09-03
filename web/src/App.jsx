import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  Check,
  CheckCircle2,
  Clipboard,
  ExternalLink,
  Github,
  Menu,
  Star,
  X,
} from 'lucide-react'
import './App.css'

const LINKS = {
  github: 'https://github.com/omertt27/Calibra',
  demo: 'https://huggingface.co/spaces/omert27/robot-dataset-health-check',
  docs: 'https://github.com/omertt27/Calibra/tree/main/docs',
  benchmarks: 'https://github.com/omertt27/Calibra#benchmark-results',
  license: 'https://github.com/omertt27/Calibra/blob/main/LICENSE',
  pypi: 'https://pypi.org/project/calibra-robotics/',
  productHunt:
    'https://www.producthunt.com/products/calibra-train-with-less-data?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-calibra-cut-robot-training-costs',
}

const INSTALL_COMMAND = 'pip install calibra-robotics'

const NAV_LINKS = [
  { href: '#benchmark', label: 'Benchmark' },
  { href: '#workflow', label: 'How it works' },
  { href: LINKS.docs, label: 'Docs' },
]

const INTEGRITY_CHECKS = [
  { label: 'Schema', detail: '165 episodes · 25,650 frames', ok: true },
  { label: 'Timestamps', detail: 'monotonic, no gaps', ok: true },
  { label: 'Action bounds', detail: 'within declared limits', ok: true },
  { label: 'Camera frames', detail: '3 episodes with decode errors', ok: false },
  { label: 'Reward signal', detail: 'present · dense', ok: true },
]

const STATS = [
  { number: '75%', label: 'less training data', note: 'at matched policy performance' },
  { number: '0.8%', label: 'difference from full-data performance', note: 'LeRobot PushT · 25% retention · 5 seeds' },
  { number: '55%', label: 'better rare-behavior preservation', note: 'vs. random coreset selection' },
]

const WORKFLOW = [
  {
    number: '01',
    name: 'Integrity',
    question: 'Can I trust this data?',
    detail: 'Detect corrupted episodes, timing problems, invalid actions, broken frames, and other structural failures.',
    command: 'calibra integrity',
  },
  {
    number: '02',
    name: 'Characterize',
    question: 'What does each episode contain?',
    detail: 'Measure quality risk, anomaly signals, coverage contribution, redundancy, success, and other episode-level characteristics.',
    command: 'calibra <path>',
  },
  {
    number: '03',
    name: 'Coverage',
    question: 'What information would I lose?',
    detail: 'Find rare behaviors, under-covered regions, and episodes that contribute meaningful diversity.',
    command: 'calibra review',
  },
  {
    number: '04',
    name: 'Decide',
    question: 'What should I train on?',
    detail: 'Keep, drop, review, annotate, or downweight episodes, or build a smaller quality-aware training set.',
    command: 'calibra prune',
  },
]

const ECOSYSTEMS = [
  { name: 'LeRobot', detail: 'v1 / v2 / v3', mark: 'LR' },
  { name: 'Isaac Lab', detail: 'HDF5', logo: 'nvidia.svg' },
  { name: 'RLDS', detail: 'TF Datasets', logo: 'tensorflow.svg' },
  { name: 'robomimic', detail: 'HDF5', mark: 'RM' },
  { name: 'Hugging Face', detail: 'Hub IDs', logo: 'huggingface.svg' },
]

function Reveal({ as: Tag = 'div', className = '', delay = 0, children, ...props }) {
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          observer.disconnect()
        }
      },
      { threshold: 0.2, rootMargin: '0px 0px -60px 0px' }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <Tag
      ref={ref}
      className={`reveal${visible ? ' in-view' : ''}${className ? ` ${className}` : ''}`}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
      {...props}
    >
      {children}
    </Tag>
  )
}

function Logo() {
  return (
    <a className="brand" href="#top" aria-label="Calibra home">
      <span>Calibra</span>
    </a>
  )
}

function StarPill() {
  const [stars, setStars] = useState(null)

  useEffect(() => {
    let live = true
    fetch('https://api.github.com/repos/omertt27/Calibra')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (live && d && typeof d.stargazers_count === 'number') setStars(d.stargazers_count)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  return (
    <a className="nav-pill" href={LINKS.github} target="_blank" rel="noreferrer">
      <Star size={13} /> {stars == null ? 'Star' : stars.toLocaleString()}
    </a>
  )
}

function HeroTerminal() {
  const [typed, setTyped] = useState('')
  const [showResults, setShowResults] = useState(false)
  const CMD = 'calibra integrity lerobot/pusht'
  const done = typed.length >= CMD.length

  useEffect(() => {
    let ticker, startTimer, resultTimer
    startTimer = setTimeout(() => {
      let i = 0
      ticker = setInterval(() => {
        i++
        setTyped(CMD.slice(0, i))
        if (i >= CMD.length) {
          clearInterval(ticker)
          resultTimer = setTimeout(() => setShowResults(true), 250)
        }
      }, 38)
    }, 800)
    return () => {
      clearTimeout(startTimer)
      clearInterval(ticker)
      clearTimeout(resultTimer)
    }
  }, [])

  return (
    <Reveal as="div" className="hero-terminal" delay={220}>
      <div className="hero-terminal-bar">
        <span><i /><i /><i /></span>
        calibra integrity
      </div>
      <div className="hero-terminal-body">
        <p className="ht-cmd">
          <span className="ht-prompt">$</span> {typed}
          {!done && <span className="ht-cursor" aria-hidden="true" />}
        </p>
        {showResults && (
          <div className="ht-results">
            {INTEGRITY_CHECKS.map((check) => (
              <p className={check.ok ? 'ht-ok' : 'ht-warn'} key={check.label}>
                {check.ok ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                {check.label}
                <span className="ht-detail">{check.detail}</span>
              </p>
            ))}
            <p className="ht-score">
              <strong>integrity score 0.98</strong> ready to train
            </p>
          </div>
        )}
      </div>
    </Reveal>
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

function StatNumber({ raw, active, delay }) {
  const target = parseFloat(raw)
  const decimals = raw.includes('.') ? 1 : 0
  const suffix = raw.replace(/[\d.]/g, '')
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    if (!active) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setDisplay(target)
      return
    }
    let raf
    const DURATION = 1500
    const startAt = performance.now() + delay
    const tick = (now) => {
      if (now < startAt) { raf = requestAnimationFrame(tick); return }
      const t = Math.min((now - startAt) / DURATION, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplay(parseFloat((eased * target).toFixed(decimals)))
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [active, target, decimals, delay])

  return <strong className="stat-number">{display.toFixed(decimals)}{suffix}</strong>
}

function StatsBar() {
  const [active, setActive] = useState(false)
  const barRef = useRef(null)

  useEffect(() => {
    const el = barRef.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setActive(true); obs.disconnect() } },
      { threshold: 0.25 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <div className="stats-bar" ref={barRef}>
      <div className="container stats-grid">
        {STATS.map((stat, index) => (
          <Reveal as="div" className="stat-item" key={stat.number} delay={index * 90}>
            <StatNumber raw={stat.number} active={active} delay={index * 90} />
            <span className="stat-label">{stat.label}</span>
            <span className="stat-note">{stat.note}</span>
          </Reveal>
        ))}
      </div>
      <div className="container">
        <p className="stats-caveat">
          Measured on LeRobot PushT. Curation gains hold up across a broader 5-seed sweep on
          3 datasets and 3 policy families (BC-MLP, ACT, Diffusion Policy).{' '}
          <a href={LINKS.benchmarks} target="_blank" rel="noreferrer">See the full benchmark table</a>.
        </p>
      </div>
    </div>
  )
}

const TAIL_COVERAGE = [
  { label: 'Calibra coreset', value: 52.0, className: 'highlight' },
  { label: 'Random baseline', value: 33.6, className: '' },
]

function CoverageGraphic() {
  return (
    <Reveal as="div" className="coverage-card">
      <div className="coverage-head">
        <div><span>Tail-behavior coverage</span><strong>41/165 episodes · LeRobot PushT</strong></div>
        <span className="coverage-score">+55% vs. random</span>
      </div>
      <div className="coverage-bars">
        {TAIL_COVERAGE.map((row) => (
          <div className={`coverage-bar-row ${row.className}`} key={row.label}>
            <span>{row.label}</span>
            <div><i style={{ '--bar-width': `${row.value}%` }} /></div>
            <strong>{row.value.toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <p className="coverage-footnote">
        Share of held-out rare-behavior episodes represented in a 41-episode coreset at 25%
        retention, from the same benchmark run shown above.
      </p>
    </Reveal>
  )
}

function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const spotlightRef = useRef(null)

  useEffect(() => {
    const handle = (e) => {
      const el = spotlightRef.current
      if (!el) return
      el.style.setProperty('--mx', `${e.clientX}px`)
      el.style.setProperty('--my', `${e.clientY}px`)
    }
    window.addEventListener('mousemove', handle, { passive: true })
    return () => window.removeEventListener('mousemove', handle)
  }, [])

  return (
    <div id="top">
      <div className="spotlight" ref={spotlightRef} />
      <nav className="nav">
        <div className="container nav-inner">
          <Logo />
          <div className="nav-links">
            {NAV_LINKS.map((link) => {
              const external = link.href.startsWith('http')
              return (
                <a
                  href={link.href}
                  key={link.label}
                  {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}
                >
                  {link.label}
                </a>
              )
            })}
          </div>
          <div className="nav-right">
            <div className="nav-badges">
              <StarPill />
              <a className="nav-pill" href={LINKS.pypi} target="_blank" rel="noreferrer">PyPI</a>
            </div>
            <a className="nav-github" href={LINKS.github} target="_blank" rel="noreferrer">
              <Github size={17} /> GitHub
            </a>
            <button
              className="nav-toggle"
              type="button"
              aria-label={menuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((v) => !v)}
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
        {menuOpen && (
          <div className="nav-menu">
            <div className="container">
              {NAV_LINKS.map((link) => {
                const external = link.href.startsWith('http')
                return (
                  <a
                    href={link.href}
                    key={link.label}
                    onClick={() => setMenuOpen(false)}
                    {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}
                  >
                    {link.label}
                  </a>
                )
              })}
              <a href={LINKS.github} target="_blank" rel="noreferrer" onClick={() => setMenuOpen(false)}>
                GitHub
              </a>
            </div>
          </div>
        )}
      </nav>

      <main>
        <header className="hero">
          <div className="container hero-grid">
            <div className="hero-copy">
              <h1>Stop wasting <span className="accent-word">GPU hours</span> on robot data.</h1>
              <p>
                Calibra is a robotics dataset intelligence layer. It audits
                integrity, measures quality and coverage, and tells you what data to keep, drop,
                review, or annotate before training, so you train on less data, preserve the
                behaviors that matter, and understand every decision before spending compute.
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                  Try the demo <ExternalLink size={16} />
                </a>
                <a className="button" href={LINKS.github} target="_blank" rel="noreferrer">
                  <Github size={16} /> View on GitHub
                </a>
              </div>
              <div className="hero-meta-row">
                <CopyCommand />
              </div>
            </div>
            <HeroTerminal />
          </div>
          <a className="scroll-cue" href="#benchmark">
            See benchmark results <ArrowDown size={15} />
          </a>
        </header>

        <StatsBar />

        <section className="research section" id="benchmark">
          <div className="container">
            <Reveal as="div" className="research-card">
              <div className="research-copy">
                <h2><strong>75% smaller</strong> dataset. Prediction error within 0.8% of full-data training.</h2>
                <p>
                  Calibra retained 41 of 165 training episodes while preserving more
                  action-space tail coverage than a random coreset, and matching the
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
                  <div><i style={{ '--bar-width': '100%' }} /></div>
                  <strong>420.67</strong>
                </div>
                <div className="benchmark-row highlight">
                  <span>Calibra <small>41 episodes</small></span>
                  <div><i style={{ '--bar-width': '99.2%' }} /></div>
                  <strong>423.92</strong>
                </div>
                <div className="benchmark-row">
                  <span>Random <small>41 episodes</small></span>
                  <div><i style={{ '--bar-width': '97.9%' }} /></div>
                  <strong>429.77</strong>
                </div>
                <div className="tail-coverage">
                  <div><span>Calibra tail coverage</span><strong>52.0%</strong></div>
                  <div><span>Random tail coverage</span><strong>33.6%</strong></div>
                </div>
                <p>Source: 5-seed retention sweep. Full-data MSE is the unfiltered 165-episode baseline.</p>
              </div>
            </Reveal>
          </div>
        </section>

        <section className="workflow section" id="workflow">
          <div className="container">
            <Reveal as="div" className="section-intro">
              <h2>Fix the data question before the model question.</h2>
              <p>Calibra turns an unknown robotics dataset into an explicit training decision.</p>
            </Reveal>
            <div className="workflow-grid">
              {WORKFLOW.map((step, index) => (
                <Reveal as="article" className="workflow-step" key={step.name} delay={index * 90}>
                  <div className="step-top">
                    <span className="step-number">{step.number}</span>
                    {index < WORKFLOW.length - 1 && <ArrowRight className="step-arrow" size={20} />}
                  </div>
                  <h3>{step.name}</h3>
                  <p className="step-question">{step.question}</p>
                  <p className="step-detail">{step.detail}</p>
                  <code>{step.command}</code>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        <section className="proof section">
          <div className="container proof-grid">
            <Reveal as="div" className="proof-copy">
              <h2>Understand every episode. Make the training decision.</h2>
              <p>
                Every Calibra decision maps back to measurable signals and the exact episodes that
                caused it. Run everything locally, inspect the reasoning, and export the result for
                your training pipeline.
              </p>
              <ul>
                <li><CheckCircle2 size={18} /> Episode-level characterization</li>
                <li><CheckCircle2 size={18} /> Explicit KEEP / DROP / REVIEW / ANNOTATE decisions</li>
                <li><CheckCircle2 size={18} /> JSONL and Parquet annotation exports</li>
                <li><CheckCircle2 size={18} /> CI-friendly, deterministic analysis</li>
                <li><CheckCircle2 size={18} /> No upload, account, or API key required</li>
              </ul>
              <a className="text-link" href={LINKS.docs} target="_blank" rel="noreferrer">Explore the commands <ArrowRight size={16} /></a>
            </Reveal>
            <CoverageGraphic />
          </div>
        </section>

        <section className="ecosystems section">
          <div className="container">
            <Reveal as="div" className="section-intro compact">
              <h2>Start with the data you already have.</h2>
            </Reveal>
            <div className="ecosystem-list">
              {ECOSYSTEMS.map((ecosystem, index) => (
                <Reveal as="div" className="ecosystem" key={ecosystem.name} delay={index * 70}>
                  <div className="ecosystem-logo">
                    {ecosystem.logo
                      ? <img src={`${import.meta.env.BASE_URL}${ecosystem.logo}`} alt="" loading="lazy" />
                      : <span>{ecosystem.mark}</span>}
                  </div>
                  <div>
                    <strong>{ecosystem.name}</strong>
                    <span>{ecosystem.detail}</span>
                  </div>
                </Reveal>
              ))}
            </div>
            <p className="support-note">
              Camera-frame integrity checks currently support HDF5 datasets and LeRobot v1 with image decoding.
            </p>
          </div>
        </section>

        <section className="demo section" id="demo">
          <div className="container demo-grid">
            <Reveal as="div" className="demo-copy">
              <h2>Inspect a public LeRobot dataset without installing anything.</h2>
              <p>
                Enter a Hub dataset ID to see integrity checks, quality findings, community
                comparisons, and a recommended keep fraction.
              </p>
              <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                Open the live demo <ExternalLink size={16} />
              </a>
            </Reveal>
            <Reveal as="a" className="demo-frame" href={LINKS.demo} target="_blank" rel="noreferrer" delay={120}>
              <div className="demo-frame-bar">
                <span><i /><i /><i /></span>
                huggingface.co/spaces/omert27/robot-dataset-health-check
              </div>
              <img src={`${import.meta.env.BASE_URL}hf-space.png`} alt="Calibra Robot Dataset Health Check on Hugging Face Spaces" />
            </Reveal>
          </div>
        </section>

        <section className="final-cta section">
          <div className="container">
            <Reveal as="div" className="cta-panel">
              <span className="cta-kicker">Before your next training run</span>
              <h2>Check the data first.</h2>
              <p>Install Calibra locally or inspect a public LeRobot dataset in the browser. Free, no account needed.</p>
              <CopyCommand large />
              <a className="button button-primary" href={LINKS.demo} target="_blank" rel="noreferrer">
                Try the Hugging Face demo <ExternalLink size={16} />
              </a>
            </Reveal>
          </div>
        </section>
      </main>

      <footer>
        <div className="container footer-inner">
          <Logo />
          <p>Train on less data. Spend less compute. Ship better policies.</p>
          <div>
            <a href={LINKS.productHunt} target="_blank" rel="noopener noreferrer" className="ph-badge">
              <img
                src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1221274&theme=light"
                alt="Calibra: Cut Robot Training Costs - Find bad data. Train on less. Save GPU. | Product Hunt"
                width={200}
                height={43}
              />
            </a>
            <a href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>
            <a href={LINKS.docs} target="_blank" rel="noreferrer">Docs</a>
            <a href={LINKS.license} target="_blank" rel="noreferrer">BSL 1.1</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
