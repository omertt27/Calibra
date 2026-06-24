import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Terminal, ShieldCheck, Cpu, Sliders, RefreshCw, BarChart3, LineChart, 
  Settings, HelpCircle, FileCheck, Layers, GitCompare, Play, Download,
  CheckCircle2, AlertTriangle, XCircle, ArrowRight, Zap, Info, Upload,
  Eye, Activity, Database, Check, Award, Flame, Code, BookOpen
} from 'lucide-react';
import { Line, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  Title,
  Tooltip,
  Legend,
  Filler
);

// Pre-loaded Robotics Datasets
const ROBOT_DATASETS = {
  aloha_mobile: {
    name: "Mobile Aloha Cabinet Curation (lerobot/aloha_mobile_cabinet)",
    episodes: 85,
    reference: "lerobot/aloha_mobile_cabinet",
    format: "HDF5 / PyTorch (14D action space)",
    jerkSpikeRate: 0.7,
    velDiscontinuity: 1.3,
    ldlj: -4.2,
    successRate: 87.0,
    outliers: [3, 19, 58],
    overallStatus: "CERTIFIED",
    flags: [
      { level: "warning", metric: "Gripper Jitter", observed: "2.1%", threshold: "<1%", msg: "Slight high-frequency jitter on gripper command channel." }
    ],
    remedyText: "Baseline reference dataset. Certified for direct policy training."
  },
  pusht: {
    name: "PushT Push-to-T Task (lerobot/pusht)",
    episodes: 100,
    reference: "lerobot/pusht",
    format: "Zarr / PyTorch (2D state space)",
    jerkSpikeRate: 4.9,
    velDiscontinuity: 16.7,
    ldlj: -6.4,
    successRate: 92.0,
    outliers: [12, 45, 87],
    overallStatus: "CERTIFIED",
    flags: [
      { level: "warning", metric: "Velocity Discontinuity", observed: "16.7%", threshold: "<20%", msg: "Minor command discontinuity at boundary transitions." }
    ],
    remedyText: "High velocity discontinuity is normal for push tasks, but check gripper transition."
  },
  custom_aloha: {
    name: "My Custom Aloha Collection (/data/my_aloha_demos/)",
    episodes: 120,
    reference: "lerobot/aloha_mobile_cabinet",
    format: "HDF5 (.h5 files)",
    jerkSpikeRate: 8.4,
    velDiscontinuity: 12.1,
    ldlj: -12.4,
    successRate: 58.0,
    outliers: [14, 22, 41, 57, 72],
    overallStatus: "REJECTED",
    flags: [
      { level: "critical", metric: "Velocity Discontinuity Rate", observed: "12.1%", threshold: "<4.0%", msg: "Severe command drops. Investigation indicates 15Hz controller lagging." },
      { level: "critical", metric: "Jerk Spike Rate", observed: "8.4%", threshold: "<5.0%", msg: "Abrupt teleoperation command corrections detected." },
      { level: "warning", metric: "Mean Trajectory Jerk (ldlj)", observed: "-12.4", threshold: ">-10.0", msg: "Action trajectories exceed jerk bounds. Policy will experience command drift." }
    ],
    remedyText: "Severe jerk anomalies detected. Curation required before training."
  }
};

const CLI_COMMANDS = {
  audit: {
    cmd: "calibra /data/my_aloha_demos/ --policy diffusion",
    desc: "Runs four kinematic and temporal analyzers over every episode, highlighting joint-space outliers.",
    output: `Scanning dataset: /data/my_aloha_demos/ (120 episodes)
Running kinematic & temporal anomaly analyzers ...
[FAIL] Velocity discontinuity rate exceeds threshold (12.1% observed, threshold 4.0%).
[FAIL] Jerk spike rate exceeds threshold (8.4% observed, threshold 5.0%).
[WARNING] Mean LDLJ score is -12.4 (warning threshold >-10.0).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA AUDIT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Episodes audited   : 120
  Mean LDLJ score    : -12.4 (FAIL)
  Jerk spike rate    : 8.4%  (FAIL)
  Vel discontinuity  : 12.1% (FAIL)
  Timestamp dropout  : 0.2%  (PASS)
  
  Status: NOT CERTIFIED (2 critical failures, 1 warning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  },
  compare: {
    cmd: "calibra compare /data/my_aloha_demos/ aloha",
    desc: "Performs evidence-backed side-by-side comparison against target reference task benchmark.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra compare — my_aloha_demos  vs.  aloha
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: lerobot/aloha_mobile_cabinet (14D · 85 episodes)
Yours:     my_aloha_demos               (120 episodes)

────────────────────────────────────────────────────────
VELOCITY DISCONTINUITY RATE
  Yours:  12.1%
  aloha   1.3%
  Delta:  +10.8%  ▲  [CRITICAL WARNING]

  Significantly rougher trajectories than reference.
  Investigate hardware communication lag or dropped packets.
────────────────────────────────────────────────────────
JERK SPIKE RATE
  Yours:  8.4%
  aloha   0.7%
  Delta:  +7.7%   ▲  [CRITICAL WARNING]

  Outliers detected by MAD analysis. Check demonstration
  boundary conditions or jerky operator command cycles.
────────────────────────────────────────────────────────

RECOMMENDED ACTIONS:
  1. Prune episode(s) 14, 22, 41, 57, 72 — jerk outliers.
  2. Apply action trajectory smoothing to recycle lag spikes.`
  },
  certify: {
    cmd: "calibra certify /data/my_aloha_demos/ --policy diffusion --strict",
    desc: "Runs automated pass/fail validation. Ideal for integrating into robot training CI/CD pipelines.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA CERTIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset  : my_aloha_demos/
  Episodes : 120
  Steps    : 93,600
  Policy   : diffusion
  Reference: aloha

  ──────────────────────────────────────────────────────────
  CERTIFICATION FAILED

  Failed Tests:
    • velocity_discontinuity: Rate 12.1% (exceeds threshold 4.0%)
    • jerk_spike_rate: Rate 8.4% (exceeds threshold 5.0%)

  Warnings:
    • ldlj: Mean LDLJ = -12.4 (threshold: >-10.0)

  ──────────────────────────────────────────────────────────
  REMEDIATION CHECKLIST
  ──────────────────────────────────────────────────────────
  1. [CRITICAL] Fix command loop latency. High velocity discontinuities 
     will prevent Diffusion Policies from training smoothly.
  2. [CRITICAL] Prune episode IDs: 14, 22, 41, 57, 72.
  3. [WARNING] Apply trajectory action smoothing to improve LDLJ score.`
  },
  prune: {
    cmd: "calibra prune /data/my_aloha_demos/ --keep 0.3 --strategy fps --out coreset.json",
    desc: "Executes coreset selection: Stage 1 filters anomalies, Stage 2 selects diverse episodes via FPS.",
    output: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA CORESET PRUNING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original episodes  : 120
  Quality failures   : 9    (removed in Stage 1)
  Redundancy pruned  : 75   (removed in Stage 2)
  Coreset size       : 36   (30.0% of original)
  Method             : quality_filter + farthest_point_sampling

  Kinematic improvements (Coreset vs Original):
    • Jerk Spike Rate        : 8.4%  → 0.8%  (▼ 90.4%)
    • Vel Discontinuity Rate : 12.1% → 2.1%  (▼ 82.6%)
    • Mean LDLJ Score        : -12.4 → -4.8  (▲ 61.2% smoother)

  To use: filter your dataset to the episode IDs in keep_episode_ids.
  Saved to coreset.json.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  },
  predict: {
    cmd: "calibra predict /data/my_aloha_demos/ --policy diffusion",
    desc: "Estimates post-training model success rate and computes GPU training time/cost savings.",
    output: `Calculating action-space coverage...
Predicting training outcomes based on reference database...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TRAINING SUCCESS PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Estimated Policy Success Rate : 58.0%  (HIGH RISK)
  
  Failure Risks:
    • 42.1% probability of gripper command drift due to joint lag.
    • 34.5% probability of action-trajectory flailing near boundaries.
    
  Coreset Recommendation:
    Coreset selection (keep=0.3) will filter quality anomalies and 
    redundancies, improving estimated success rate to 88.0%.
    
  GPU Hours Saved estimate: 24.5 hours (worth ~$196.00 on A100-80G).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [cliTab, setCliTab] = useState('audit');
  
  // Settings Workspace States
  const [selectedDatasetKey, setSelectedDatasetKey] = useState('custom_aloha');
  const [keepRatio, setKeepRatio] = useState(0.30);
  const [pruningStrategy, setPruningStrategy] = useState('fps');
  const [applyQualityFilter, setApplyQualityFilter] = useState(true);
  const [applySmoothing, setApplySmoothing] = useState(false);
  const [isCurationRunning, setIsCurationRunning] = useState(false);
  const [curationProgress, setCurationProgress] = useState(100);

  // Live Telemetry Corruption Simulator
  const [injectedJitter, setInjectedJitter] = useState(0);
  const [injectedDrops, setInjectedDrops] = useState(0);

  // Selected Episode Lock state
  const [selectedEpisodeId, setSelectedEpisodeId] = useState(14); // Defaults to Episode 14 (an anomaly)
  
  // Preload dataset data
  const currentDataset = useMemo(() => {
    return ROBOT_DATASETS[selectedDatasetKey];
  }, [selectedDatasetKey]);

  // Generate 2D Behavioral Embedding Space (Deterministically LCG-seeded)
  const embeddingPoints = useMemo(() => {
    let seed = 12345;
    const lcgRandom = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };

    const points = [];
    const clusters = [
      { cx: 32, cy: 38, r: 10, label: "Reaching trajectory" },
      { cx: 68, cy: 58, r: 12, label: "Gripper grasp phase" },
      { cx: 50, cy: 26, r: 8, label: "Cabinet alignment" },
      { cx: 22, cy: 70, r: 9, label: "Retracting phase" }
    ];

    const totalCount = 150;
    const anomalyIndices = [14, 22, 41, 57, 72, 88, 102, 115, 134, 142];

    // Determine baseline multipliers depending on the chosen dataset key
    const jerkMultiplier = selectedDatasetKey === 'custom_aloha' ? 1.3 : selectedDatasetKey === 'pusht' ? 0.8 : 0.15;
    const dropsMultiplier = selectedDatasetKey === 'custom_aloha' ? 1.2 : selectedDatasetKey === 'pusht' ? 1.6 : 0.2;

    for (let i = 0; i < totalCount; i++) {
      const cluster = clusters[i % clusters.length];
      const angle = lcgRandom() * Math.PI * 2;
      const distance = lcgRandom() * cluster.r;
      const x = cluster.cx + Math.cos(angle) * distance;
      const y = cluster.cy + Math.sin(angle) * distance;
      
      const isBaseAnomaly = anomalyIndices.includes(i + 1);

      // Generate base telemetry levels
      let baseJerk = (isBaseAnomaly ? 5.5 : 0.3) + lcgRandom() * 2.0;
      let baseDrops = (isBaseAnomaly ? 6.0 : 0.4) + lcgRandom() * 3.0;

      // Adjust based on the selected dataset profile
      baseJerk *= jerkMultiplier;
      baseDrops *= dropsMultiplier;

      // Scale slider corruption offsets slightly differently per point to simulate realistic sensor variations
      const multiplier = 0.6 + lcgRandom() * 0.8;
      const finalJerk = baseJerk + injectedJitter * multiplier;
      const finalDrops = baseDrops + injectedDrops * multiplier;

      // If trajectory smoothing is active, it suppresses telemetry noise
      const smoothedJerk = applySmoothing ? finalJerk * 0.12 : finalJerk;
      const smoothedDrops = applySmoothing ? finalDrops * 0.20 : finalDrops;

      // Quality anomaly flags triggered if jerk > 5.0% or drop velocity > 4.0%
      const isQualityAnomaly = smoothedJerk > 5.0 || smoothedDrops > 4.0;

      points.push({
        id: i + 1,
        x: parseFloat(x.toFixed(2)),
        y: parseFloat(y.toFixed(2)),
        isQualityAnomaly,
        jerkRate: parseFloat(smoothedJerk.toFixed(1)),
        velDiscontinuity: parseFloat(smoothedDrops.toFixed(1)),
        entropy: parseFloat((lcgRandom() * 3 + 2.5).toFixed(2)),
        clusterName: cluster.label
      });
    }

    // Determine Farthest Point Sampling order
    const fpsOrder = [];
    const visited = new Set();
    fpsOrder.push(0);
    visited.add(0);

    while (fpsOrder.length < totalCount) {
      let maxDist = -1;
      let nextIdx = -1;
      for (let i = 0; i < totalCount; i++) {
        if (visited.has(i)) continue;
        let minDistToVisited = Infinity;
        for (const v of fpsOrder) {
          const dx = points[i].x - points[v].x;
          const dy = points[i].y - points[v].y;
          const dist = dx * dx + dy * dy;
          if (dist < minDistToVisited) {
            minDistToVisited = dist;
          }
        }
        if (minDistToVisited > maxDist) {
          maxDist = minDistToVisited;
          nextIdx = i;
        }
      }
      fpsOrder.push(nextIdx);
      visited.add(nextIdx);
    }

    // Assign FPS rank to each point
    fpsOrder.forEach((pointIdx, rank) => {
      points[pointIdx].fpsRank = rank;
    });

    return points;
  }, [selectedDatasetKey, injectedJitter, injectedDrops, applySmoothing]);

  // Compute active coreset based on parameters
  const curationResult = useMemo(() => {
    // 1. Filter anomalies if filter active
    let pool = [...embeddingPoints];
    const anomaliesCount = pool.filter(p => p.isQualityAnomaly).length;
    
    let filteredCount = 0;
    if (applyQualityFilter) {
      pool = pool.filter(p => !p.isQualityAnomaly);
      filteredCount = anomaliesCount;
    }

    // 2. Select diverse coreset
    // Sort by rank: Farthest Point Sampling rank
    pool.sort((a, b) => a.fpsRank - b.fpsRank);

    const targetK = Math.max(5, Math.floor(keepRatio * embeddingPoints.length));
    const keptSet = new Set();
    
    // Greedy select top targetK passing points
    for (let i = 0; i < pool.length && keptSet.size < targetK; i++) {
      keptSet.add(pool[i].id);
    }

    const coresetPoints = embeddingPoints.filter(p => keptSet.has(p.id));
    const prunedPoints = embeddingPoints.filter(p => !keptSet.has(p.id) && (!applyQualityFilter || !p.isQualityAnomaly));
    const rejectedPoints = applyQualityFilter ? embeddingPoints.filter(p => p.isQualityAnomaly) : [];

    return {
      coreset: coresetPoints,
      pruned: prunedPoints,
      rejected: rejectedPoints,
      totalKept: keptSet.size,
      totalRedundant: prunedPoints.length,
      totalRejected: rejectedPoints.length
    };
  }, [embeddingPoints, keepRatio, applyQualityFilter]);

  // Interactive Embedding Canvas
  const canvasRef = useRef(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const drawEmbeddingSpace = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw Grid mesh background
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    const gridStep = 40;
    for (let x = 0; x < width; x += gridStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Canvas scaling parameters
    const padding = 30;
    const minX = 0, maxX = 90;
    const minY = 0, maxY = 90;

    const scaleX = (x) => padding + ((x - minX) / (maxX - minX)) * (width - padding * 2);
    const scaleY = (y) => height - padding - ((y - minY) / (maxY - minY)) * (height - padding * 2);

    // 1. Draw diversity-pruned points (faded blue dots)
    curationResult.pruned.forEach(p => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      ctx.fillStyle = 'rgba(148, 163, 184, 0.25)'; // slate-400 faded
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // 2. Draw quality-rejected points (red X or red dots)
    curationResult.rejected.forEach(p => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)'; // red-500
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 4, cy - 4);
      ctx.lineTo(cx + 4, cy + 4);
      ctx.moveTo(cx + 4, cy - 4);
      ctx.lineTo(cx - 4, cy + 4);
      ctx.stroke();
    });

    // 3. Draw coreset-kept points (gold glowing circles)
    curationResult.coreset.forEach(p => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      
      // Draw outer ring glow
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.25)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.stroke();

      // Fill inner circle
      ctx.fillStyle = '#f59e0b'; // Calibra gold
      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Highlight hovered point
    if (hoveredPoint) {
      const cx = scaleX(hoveredPoint.x);
      const cy = scaleY(hoveredPoint.y);
      
      ctx.strokeStyle = '#38bdf8'; // sky blue pulse
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw persistent selected episode (highlighted with double cyan ring)
    if (selectedEpisodeId) {
      const p = embeddingPoints.find(ep => ep.id === selectedEpisodeId);
      if (p) {
        const cx = scaleX(p.x);
        const cy = scaleY(p.y);
        
        ctx.strokeStyle = '#06b6d4'; // cyan-500
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  };

  useEffect(() => {
    drawEmbeddingSpace();
  }, [curationResult, hoveredPoint, selectedEpisodeId]);

  const handleCanvasMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;
    const minX = 0, maxX = 90;
    const minY = 0, maxY = 90;

    const scaleX = (x) => padding + ((x - minX) / (maxX - minX)) * (width - padding * 2);
    const scaleY = (y) => height - padding - ((y - minY) / (maxY - minY)) * (height - padding * 2);

    let match = null;
    let minDist = 12; // hover sensitivity pixels

    embeddingPoints.forEach(p => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      const dx = mx - cx;
      const dy = my - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) {
        minDist = dist;
        match = p;
      }
    });

    setHoveredPoint(match);
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;
    const minX = 0, maxX = 90;
    const minY = 0, maxY = 90;

    const scaleX = (x) => padding + ((x - minX) / (maxX - minX)) * (width - padding * 2);
    const scaleY = (y) => height - padding - ((y - minY) / (maxY - minY)) * (height - padding * 2);

    let match = null;
    let minDist = 14; // click sensitivity pixels

    embeddingPoints.forEach(p => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      const dx = mx - cx;
      const dy = my - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) {
        minDist = dist;
        match = p;
      }
    });

    if (match) {
      setSelectedEpisodeId(match.id);
      setCliTab('audit'); // Switch to trajectory audit tab to show immediate details
    }
  };
  
  // Keyboard ergonomics event listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (document.activeElement?.tagName === 'INPUT' && document.activeElement?.type === 'text') {
        return;
      }

      // 1. Sub-panel selector shortcuts (keys 1 - 4)
      if (activeTab === 'simulator') {
        if (e.key === '1') setCliTab('audit');
        if (e.key === '2') setCliTab('coreset');
        if (e.key === '3') setCliTab('compare');
        if (e.key === '4') setCliTab('watch');
      }

      // 2. Main Page Tab shortcuts (o/O for Overview, w/W for Workspace)
      if (e.key.toLowerCase() === 'o') setActiveTab('landing');
      if (e.key.toLowerCase() === 'w') setActiveTab('simulator');

      // 3. Arrow navigation for episodes (ArrowRight next, ArrowLeft previous)
      if (e.key === 'ArrowRight') {
        e.preventDefault(); // Prevent page scroll
        setSelectedEpisodeId(prev => {
          if (prev === null) return 1;
          return prev >= 150 ? 1 : prev + 1;
        });
        setCliTab('audit'); // Switch to trajectory plot to inspect immediately
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault(); // Prevent page scroll
        setSelectedEpisodeId(prev => {
          if (prev === null) return 150;
          return prev <= 1 ? 150 : prev - 1;
        });
        setCliTab('audit');
      }

      // 4. Escape key to clear locked episode
      if (e.key === 'Escape') {
        setSelectedEpisodeId(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeTab]);

  // Mock upload logic
  const handleDatasetUpload = (e) => {
    e.preventDefault();
    setIsCurationRunning(true);
    setCurationProgress(0);
    
    // Simulate training progress or file scanning
    const interval = setInterval(() => {
      setCurationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsCurationRunning(false);
          setSelectedDatasetKey('custom_aloha');
          return 100;
        }
        return prev + 25;
      });
    }, 300);
  };

  // Execute pruning simulation with active loading animation
  const triggerCurationRun = () => {
    setIsCurationRunning(true);
    setCurationProgress(0);
    
    const interval = setInterval(() => {
      setCurationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsCurationRunning(false);
          return 100;
        }
        return prev + 10;
      });
    }, 100);
  };

  const getScoreColor = (score) => {
    if (score >= 85) return 'var(--success)';
    if (score >= 70) return 'var(--warning)';
    return 'var(--danger)';
  };

  const getStatusBadgeClass = (status) => {
    if (status === 'CERTIFIED') return 'status-badge-success';
    if (status === 'PROVISIONALLY CERTIFIED') return 'status-badge-warning';
    return 'status-badge-danger';
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

  // Calculate dynamic success rate based on the coreset selection and parameters
  const estimatedSuccessRate = useMemo(() => {
    const coreset = curationResult.coreset;
    const coresetSize = coreset.length;
    
    // Average metrics in the selected coreset
    const avgJerk = coreset.reduce((sum, p) => sum + p.jerkRate, 0) / (coresetSize || 1);
    const avgDisc = coreset.reduce((sum, p) => sum + p.velDiscontinuity, 0) / (coresetSize || 1);
    
    // Count remaining anomalies in the curated coreset
    const anomaliesInCoreset = coreset.filter(p => p.isQualityAnomaly).length;
    
    let score = 94;
    
    // 1. Heavy penalty for containing quality anomalies
    score -= anomaliesInCoreset * 5.0;
    
    // 2. Penalty for generic noise levels
    if (avgJerk > 4.0) score -= (avgJerk - 4.0) * 3.0;
    if (avgDisc > 3.0) score -= (avgDisc - 3.0) * 2.0;
    
    // 3. Penalty for data starvation if too many episodes are filtered out
    if (coresetSize < 35) {
      score -= (35 - coresetSize) * 0.9;
    }
    
    // 4. Boost for active smoothing
    if (applySmoothing) {
      score += 5;
    }
    
    return Math.max(12, Math.min(96, Math.round(score)));
  }, [curationResult, applySmoothing]);

  // Calculate active dataset-wide average metrics
  const datasetMetrics = useMemo(() => {
    const total = embeddingPoints.length;
    if (total === 0) return { jerkRate: 0, velDiscontinuity: 0, ldlj: 0, anomalyRate: 0 };
    
    const sumJerk = embeddingPoints.reduce((sum, p) => sum + p.jerkRate, 0);
    const sumDisc = embeddingPoints.reduce((sum, p) => sum + p.velDiscontinuity, 0);
    const anomalies = embeddingPoints.filter(p => p.isQualityAnomaly).length;
    
    const avgJerk = sumJerk / total;
    const avgDisc = sumDisc / total;
    const avgLdlj = -4.0 - avgJerk * 0.8;
    
    return {
      jerkRate: parseFloat(avgJerk.toFixed(1)),
      velDiscontinuity: parseFloat(avgDisc.toFixed(1)),
      ldlj: parseFloat(avgLdlj.toFixed(1)),
      anomalyRate: parseFloat(((anomalies / total) * 100).toFixed(1))
    };
  }, [embeddingPoints]);

  // Dynamic trajectory simulator based on the selected episode
  const activeEpisodeForTrajectory = useMemo(() => {
    const ep = embeddingPoints.find(p => p.id === selectedEpisodeId);
    return ep || embeddingPoints[0];
  }, [embeddingPoints, selectedEpisodeId]);

  const dynamicTrajectoryData = useMemo(() => {
    if (!activeEpisodeForTrajectory) return { raw: [], smoothed: [], anomalySteps: [] };
    const ep = activeEpisodeForTrajectory;
    
    // Deterministic random generator using episode ID as seed
    let seed = ep.id * 1000 + 7;
    const lcg = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    
    const length = 100;
    const raw = [];
    const anomalySteps = [];
    
    for (let i = 0; i < length; i++) {
      // Base trajectory representing robot arm angular command (smooth sine)
      const base = Math.sin(i / 15) * 1.4 + Math.cos(i / 8) * 0.4;
      
      // Inject anomalies proportional to the episode's jerkRate
      // The higher the rate, the higher the probability of spikes
      const spikeThreshold = 1.0 - (ep.jerkRate / 30);
      const isAnomalyStep = ep.isQualityAnomaly && lcg() > Math.max(0.6, spikeThreshold) && i > 10 && i < 90;
      
      let noise = (lcg() - 0.5) * 0.08; // minor jitter
      
      if (isAnomalyStep) {
        const direction = lcg() > 0.5 ? 1 : -1;
        const magnitude = 0.8 + (ep.jerkRate / 6.0) + lcg() * 0.6;
        noise += direction * magnitude;
        anomalySteps.push(i);
      }
      
      // Inject a big velocity discontinuity step if the packet drop rate is high
      if (ep.isQualityAnomaly && i === 50 && ep.velDiscontinuity > 4.5) {
        noise += 1.5;
        anomalySteps.push(i);
      }
      
      raw.push(base + noise);
    }
    
    // Smooth the raw trajectory (Savitzky-Golay / rolling average representation)
    const smoothed = [];
    for (let i = 0; i < length; i++) {
      let sum = 0;
      let count = 0;
      const window = 3; // 7-point window
      for (let j = -window; j <= window; j++) {
        if (i + j >= 0 && i + j < length) {
          sum += raw[i + j];
          count++;
        }
      }
      smoothed.push(sum / count);
    }
    
    return { raw, smoothed, anomalySteps };
  }, [activeEpisodeForTrajectory]);

  const activeFlags = useMemo(() => {
    const flags = [];
    const ep = activeEpisodeForTrajectory;
    if (!ep) return [];
    
    if (ep.jerkRate > 5.0) {
      flags.push({
        level: "critical",
        metric: "Jerk Spike Rate",
        observed: `${ep.jerkRate}%`,
        threshold: "< 5.0%",
        msg: `High-frequency joint spikes detected. Telemetry flags abrupt operator corrections in movement segment '${ep.clusterName}'.`
      });
    }
    if (ep.velDiscontinuity > 4.0) {
      flags.push({
        level: "critical",
        metric: "Velocity Discontinuity",
        observed: `${ep.velDiscontinuity}%`,
        threshold: "< 4.0%",
        msg: `Severe transmission gaps. Observability logs indicate actuator command packet losses or control loop latency.`
      });
    }
    const epLdlj = -4.0 - ep.jerkRate * 0.8;
    if (epLdlj < -10.0) {
      flags.push({
        level: "warning",
        metric: "Mean Trajectory Jerk (ldlj)",
        observed: epLdlj.toFixed(1),
        threshold: "> -10.0",
        msg: "Action trajectories exceed standard smooth bounds. Policy training may experience significant command drift."
      });
    }
    return flags;
  }, [activeEpisodeForTrajectory]);

  const trajectoryLabels = Array.from({ length: 100 }, (_, i) => `Step ${i}`);

  const trajectoryChartData = {
    labels: trajectoryLabels,
    datasets: [
      {
        label: applySmoothing ? 'Smoothed Joint Velocity (Savitzky-Golay Filter)' : 'Raw Teleoperation command trace',
        data: applySmoothing ? dynamicTrajectoryData.smoothed : dynamicTrajectoryData.raw,
        borderColor: applySmoothing ? 'rgba(16, 185, 129, 1)' : 'rgba(245, 158, 11, 1)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: (context) => {
          if (applySmoothing) return 0;
          const idx = context.dataIndex;
          if (dynamicTrajectoryData.anomalySteps.includes(idx)) return 6;
          return 0;
        },
        pointBackgroundColor: 'rgba(239, 68, 68, 1)',
        pointBorderColor: '#fff',
        pointBorderWidth: 1.5,
        tension: 0.2,
      }
    ]
  };

  const trajectoryChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#64748b' } },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b' },
        title: { display: true, text: 'Angular Velocity (rad/s)', color: '#94a3b8' }
      }
    }
  };

  // Live Watch Streaming Simulation
  const [watchLog, setWatchLog] = useState([
    { time: "17:09:12", msg: "Streaming loop initialized: calibra watch --stream" },
    { time: "17:09:14", msg: "Episode 101 finished. Steps: 820. LDLJ: -4.1. Certified" },
    { time: "17:09:18", msg: "Episode 102 finished. Steps: 750. LDLJ: -4.5. Certified" }
  ]);

  const [streamVelocity, setStreamVelocity] = useState(0.8);

  useEffect(() => {
    if (activeTab !== 'simulator') return;
    
    // Simulate real-time streaming updates
    const interval = setInterval(() => {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const isAnomaly = Math.random() > 0.75;
      
      let msg = "";
      if (isAnomaly) {
        msg = `Episode ${Math.floor(Math.random() * 50 + 103)} finished. Gripper command jerk spike flagged! Delta: +8.2 rad/s²`;
      } else {
        msg = `Episode ${Math.floor(Math.random() * 50 + 103)} finished. Steps: ${Math.floor(Math.random() * 100 + 750)}. LDLJ: -3.8. Certified`;
      }

      setWatchLog(prev => [
        { time: timeStr, msg },
        ...prev.slice(0, 6)
      ]);

      setStreamVelocity(Math.sin(Date.now() / 1000) * 0.5 + 1.2);
    }, 4500);

    return () => clearInterval(interval);
  }, [activeTab]);

  return (
    <div style={{ position: 'relative', minHeight: '100vh', paddingBottom: '100px' }}>
      <div className="grid-overlay" />
      <div className="bg-glow-orb bg-glow-top-left" />
      <div className="bg-glow-orb bg-glow-bottom-right" />

      {/* Navigation Header */}
      <header style={{ borderBottom: '1px solid var(--border-color)', backdropFilter: 'blur(16px)', sticky: 'top', top: 0, zIndex: 10, background: 'rgba(4, 8, 21, 0.8)' }}>
        <div className="container" style={{ height: '76px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="logo-container" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('landing')}>
            <img src="/logo-icon.svg" alt="Calibra" style={{ height: '36px', width: '36px' }} />
            <div className="logo-wordmark">Calibra</div>
            <span style={{ height: '16px', width: '1px', background: 'rgba(255,255,255,0.15)' }} />
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500, letterSpacing: '0.05em' }}>DATASET OBSERVABILITY</span>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              onClick={() => setActiveTab('landing')} 
              className={`tab-btn ${activeTab === 'landing' ? 'tab-btn-active' : ''}`}
              style={{ width: 'auto', padding: '8px 18px' }}
            >
              <BookOpen size={15} /> Overview
            </button>
            <button 
              onClick={() => setActiveTab('simulator')} 
              className={`tab-btn ${activeTab === 'simulator' ? 'tab-btn-active' : ''}`}
              style={{ width: 'auto', padding: '8px 18px' }}
            >
              <Activity size={15} /> Control Panel & Curation Workspace
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="container" style={{ marginTop: '48px' }}>
        
        {/* OVERVIEW / LANDING TAB */}
        {activeTab === 'landing' && (
          <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '80px' }}>
            
            {/* Hero Section */}
            <section style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '26px', marginTop: '20px' }}>
              <div className="highlight-pill">
                <Zap size={14} /> Saving up to 70% GPU Curation Costs
              </div>

              <h1 style={{ fontSize: '64px', fontWeight: 800, maxWidth: '900px', lineHeight: '1.1', background: 'linear-gradient(to right, #ffffff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Dataset Observability & Coreset Curation <br />built for Robotics
              </h1>

              <p style={{ fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '720px', margin: '0 auto', lineHeight: '1.7' }}>
                Prune redundant robot demonstrations, audit kinematic anomalies, and certify your training datasets before you waste expensive GPU hours.
              </p>

              <div style={{ display: 'flex', gap: '16px', marginTop: '12px' }}>
                <button onClick={() => setActiveTab('simulator')} className="btn-primary">
                  <Play size={16} fill="currentColor" /> Open Curation Panel
                </button>
                <a href="#cli-section" className="btn-secondary" style={{ textDecoration: 'none' }} onClick={(e) => { e.preventDefault(); document.getElementById('cli-section')?.scrollIntoView({ behavior: 'smooth' }); }}>
                  Explore SDK / CLI
                </a>
              </div>
            </section>

            {/* Impact Metric Cards */}
            <section style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
              <div className="glass-card" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '40px', fontWeight: 800, color: 'var(--brand-gold)' }}>70%</span>
                <strong style={{ fontSize: '15px' }}>GPU Training Time Saved</strong>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Prune redundant, non-informative movements that contribute zero gradient signal.</p>
              </div>
              <div className="glass-card" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '40px', fontWeight: 800, color: 'var(--success)' }}>100%</span>
                <strong style={{ fontSize: '15px' }}>Automated Quality Certification</strong>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Eliminate episodes containing command drops, actuators stuck, or frame drops.</p>
              </div>
              <div className="glass-card" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '40px', fontWeight: 800, color: '#38bdf8' }}>5x</span>
                <strong style={{ fontSize: '15px' }}>Faster Policy Validation</strong>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Run programmatic diagnostics inside continuous integration loops.</p>
              </div>
            </section>

            {/* Core Columns */}
            <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(245, 158, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--brand-gold)' }}>
                  <Sliders size={24} />
                </div>
                <h3>Programmatic Kinematic Auditing</h3>
                <p style={{ fontSize: '14.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  Robot imitation learning fails silently on noisy trajectories. Calibra checks demonstration logs for joint-space velocity spikes, command jitters, and physical delays, ensuring your model learns only smooth, reproducible trajectories.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--brand-gold)', fontWeight: 600 }}>
                  <CheckCircle2 size={15} /> Evaluates LDLJ & bootstrap confidence parameters.
                </div>
              </div>

              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(14, 165, 233, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0ea5e9' }}>
                  <Layers size={24} />
                </div>
                <h3>Symmetric Coreset Selection</h3>
                <p style={{ fontSize: '14.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  Avoid duplicate demonstration training. Using Stage 1 Kinematic filters and Stage 2 Farthest-Point Sampling, Calibra extracts the most behaviorally diverse demonstrations. Select subsets that maintain maximum state-space coverage.
                </p>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#0ea5e9', fontWeight: 600 }}>
                  <CheckCircle2 size={15} /> Selects optimal subsets with mathematical rigor.
                </div>
              </div>
            </section>

            {/* CLI Command Shell Sandbox */}
            <section id="cli-section" style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
              <div style={{ textAlign: 'left' }}>
                <h2>Robotics Curation SDK & CLI Sandbox</h2>
                <p style={{ fontSize: '14.5px', color: 'var(--text-secondary)' }}>A developer-first command-line tool written in Python. Simple to integrate into training scripts and robot collectors.</p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '32px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {Object.keys(CLI_COMMANDS).map((cmdKey) => (
                    <button
                      key={cmdKey}
                      onClick={() => setCliTab(cmdKey)}
                      className={`tab-btn ${cliTab === cmdKey ? 'tab-btn-active' : ''}`}
                      style={{ justifyContent: 'flex-start', padding: '12px 18px' }}
                    >
                      <Terminal size={14} />
                      <span style={{ textTransform: 'capitalize' }}>calibra {cmdKey}</span>
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
                    <span className="terminal-title">calibra-cli --zsh</span>
                    <span style={{ width: '40px' }} />
                  </div>
                  <div className="terminal-body" style={{ maxHeight: '380px', overflowY: 'auto' }}>
                    <div className="terminal-line">
                      <span className="terminal-prompt">$</span>
                      <span style={{ color: '#fff', fontWeight: 600 }}>{CLI_COMMANDS[cliTab].cmd}</span>
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px' }}>
                      # {CLI_COMMANDS[cliTab].desc}
                    </div>
                    <pre style={{ color: '#e2e8f0', fontFamily: 'var(--font-mono)', fontSize: '13px', whiteSpace: 'pre-wrap' }}>
                      {CLI_COMMANDS[cliTab].output}
                    </pre>
                  </div>
                </div>
              </div>
            </section>

          </div>
        )}

        {/* WORKSPACE & CURATION PANEL TAB */}
        {activeTab === 'simulator' && (
          <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            
            {/* Top Workspace Config Row */}
            <div className="glass-card" style={{ display: 'flex', flexWrap: 'wrap', gap: '28px', alignItems: 'center', justifyContent: 'space-between', padding: '24px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left' }}>
                <label style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Active Demonstration Dataset</label>
                <select 
                  value={selectedDatasetKey} 
                  onChange={(e) => setSelectedDatasetKey(e.target.value)}
                  className="glass-input"
                  style={{ minWidth: '320px', background: '#080f25' }}
                >
                  <option value="custom_aloha">My Custom Aloha Collection (/data/my_aloha_demos/)</option>
                  <option value="aloha_mobile">Mobile Aloha Reference (lerobot/aloha_mobile_cabinet)</option>
                  <option value="pusht">PushT Demonstration Dataset (lerobot/pusht)</option>
                </select>
              </div>

              {/* Upload new dataset logs */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>or analyze your own demos:</span>
                <form onSubmit={handleDatasetUpload}>
                  <button 
                    disabled={isCurationRunning}
                    className="btn-secondary" 
                    style={{ fontSize: '13.5px', padding: '10px 18px', display: 'flex', alignItems: 'center', gap: '8px' }}
                  >
                    <Upload size={14} /> 
                    {isCurationRunning && curationProgress < 100 ? `Analyzing [${curationProgress}%]` : "Upload .H5 / .Zarr Trajectories"}
                  </button>
                </form>
              </div>

              <div style={{ display: 'flex', gap: '24px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>TOTAL EPISODES</div>
                  <div style={{ fontSize: '18px', fontWeight: 700 }}>{currentDataset.episodes}</div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>OUTLIERS DETECTED</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: currentDataset.outliers.length > 3 ? 'var(--danger)' : 'var(--text-primary)' }}>
                    {currentDataset.outliers.length}
                  </div>
                </div>
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.08)' }} />
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>FORMAT SPEC</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--brand-gold)' }}>{currentDataset.format}</div>
                </div>
              </div>
            </div>

            {/* Split layout: sidebar inputs vs workspace tabs */}
            <div className="dashboard-grid">
              
              {/* Left Panel: Settings Controls */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                
                {/* Accuracy Gating / Success Prediction */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', padding: '24px' }}>
                  <h3 style={{ fontSize: '14.5px', color: 'var(--text-secondary)', fontWeight: 600 }}>Estimated Policy Success Rate</h3>
                  
                  {/* Gauge updates based on dataset selection and parameters */}
                  {renderGauge(estimatedSuccessRate)}
                  
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>PREDICTION MODEL</div>
                    <div style={{ fontSize: '14px', fontWeight: 700, color: 'rgba(255,255,255,0.9)' }}>
                      Diffusion Policy (14D space)
                    </div>
                  </div>
                </div>

                {/* Live Hardware Corruption Simulator (calibra corrupt) */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left', padding: '24px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--brand-gold)' }}>
                    <Flame size={16} /> Live Hardware Corruption (calibra corrupt)
                  </h3>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    Inject telemetry drops or control noise to simulate physical actuators degrading.
                  </p>
                  
                  {/* Slider 1: Inject Jitter (accelerometer/actuator noise) */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Inject Actuator Jitter:</span>
                      <strong style={{ color: 'var(--brand-gold)', fontSize: '14px' }}>+{injectedJitter}%</strong>
                    </div>
                    <input 
                      type="range" 
                      min="0" 
                      max="15" 
                      step="1" 
                      value={injectedJitter} 
                      onChange={(e) => setInjectedJitter(parseInt(e.target.value))}
                      style={{ width: '100%' }}
                    />
                    <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                      Simulates high-frequency control loop noise (adds jerk).
                    </span>
                  </div>

                  {/* Slider 2: Inject Packet Drops (lag/control delays) */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Inject Packet Drops (Lag):</span>
                      <strong style={{ color: 'var(--brand-gold)', fontSize: '14px' }}>+{injectedDrops}%</strong>
                    </div>
                    <input 
                      type="range" 
                      min="0" 
                      max="25" 
                      step="1" 
                      value={injectedDrops} 
                      onChange={(e) => setInjectedDrops(parseInt(e.target.value))}
                      style={{ width: '100%' }}
                    />
                    <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                      Simulates command dropouts (causes discontinuity).
                    </span>
                  </div>
                </div>

                {/* Hyperparameters Controls */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '18px', textAlign: 'left' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Sliders size={16} className="text-amber-500" /> Curation Hyperparameters
                  </h3>

                  {/* Keep Ratio Slider */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Coreset Keep Ratio:</span>
                      <strong style={{ color: 'var(--brand-gold)', fontSize: '15px' }}>
                        {(keepRatio * 100).toFixed(0)}%
                      </strong>
                    </div>
                    <input 
                      type="range" 
                      min="0.10" 
                      max="0.90" 
                      step="0.05" 
                      value={keepRatio} 
                      onChange={(e) => setKeepRatio(parseFloat(e.target.value))}
                      style={{ width: '100%' }}
                    />
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                      Yields {Math.max(5, Math.floor(keepRatio * 150))} episodes out of 150 embeddings.
                    </span>
                  </div>

                  {/* Curation Strategy */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Diversity Selector:</label>
                    <select 
                      value={pruningStrategy} 
                      onChange={(e) => setPruningStrategy(e.target.value)}
                      className="glass-input"
                      style={{ background: '#080f25' }}
                    >
                      <option value="fps">Farthest Point Sampling (O(N*K) Greedy)</option>
                      <option value="entropy">Entropy-Biased selection</option>
                      <option value="influence">Influence Estimation (State Novelty)</option>
                    </select>
                  </div>

                  {/* Switch toggles */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '6px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontSize: '13.5px' }}>
                      <input 
                        type="checkbox" 
                        checked={applyQualityFilter} 
                        onChange={(e) => setApplyQualityFilter(e.target.checked)}
                        style={{ accentColor: 'var(--brand-gold)', transform: 'scale(1.1)' }} 
                      />
                      <div>
                        <strong>Stage 1 Quality Filter</strong>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>Auto-prunes kinematics outlier spikes</div>
                      </div>
                    </label>

                    <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontSize: '13.5px' }}>
                      <input 
                        type="checkbox" 
                        checked={applySmoothing} 
                        onChange={(e) => setApplySmoothing(e.target.checked)}
                        style={{ accentColor: 'var(--brand-gold)', transform: 'scale(1.1)' }} 
                      />
                      <div>
                        <strong>Apply Action Trajectory Smoothing</strong>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>Savitzky-Golay filters on high frequency noise</div>
                      </div>
                    </label>
                  </div>

                  <button 
                    onClick={triggerCurationRun}
                    className="btn-primary" 
                    style={{ width: '100%', justifyContent: 'center', marginTop: '8px' }}
                  >
                    <RefreshCw size={16} className={isCurationRunning ? "animate-spin" : ""} />
                    {isCurationRunning ? "Pruning Dataset..." : "Run Curation Pipeline"}
                  </button>
                </div>

                {/* Technical check summary */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '12px', textAlign: 'left', fontSize: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--warning)' }}>
                    <AlertTriangle size={15} />
                    <strong>Dataset Observability Check</strong>
                  </div>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    {curationResult.totalRejected > 0 && applyQualityFilter
                      ? `Coreset curation applied. Automatically pruned ${curationResult.totalRejected} kinematic anomaly episodes. Saved ${((curationResult.totalRejected * 2.5)).toFixed(1)} hours of training compute.`
                      : curationResult.totalRejected > 0
                      ? `CRITICAL WARNING: Training on raw demonstrations containing ${curationResult.totalRejected} anomalies will result in model drift and action flailing.`
                      : "Telemetry benchmarks align correctly. Standard dataset ready for imitation learning."}
                  </p>
                </div>

              </div>

              {/* Right Panel: Workspace Tabs & Visualizations */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                
                {/* Dashboard Tabs Selector */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div className="tab-container" style={{ marginBottom: 0 }}>
                    <button 
                      onClick={() => setCliTab('audit')} 
                      className={`tab-btn ${cliTab === 'audit' ? 'tab-btn-active' : ''}`}
                    >
                      <Activity size={14} /> Kinematic Trajectory Audit
                    </button>
                    <button 
                      onClick={() => setCliTab('coreset')} 
                      className={`tab-btn ${cliTab === 'coreset' ? 'tab-btn-active' : ''}`}
                    >
                      <Layers size={14} /> Behavioral Coreset Space (2D)
                    </button>
                    <button 
                      onClick={() => setCliTab('compare')} 
                      className={`tab-btn ${cliTab === 'compare' ? 'tab-btn-active' : ''}`}
                    >
                      <GitCompare size={14} /> Compare to Benchmarks
                    </button>
                    <button 
                      onClick={() => setCliTab('watch')} 
                      className={`tab-btn ${cliTab === 'watch' ? 'tab-btn-active' : ''}`}
                    >
                      <Eye size={14} /> Real-time watch stream
                    </button>
                  </div>

                  {/* TAB 1: KINEMATIC TRAJECTORY AUDIT */}
                  {cliTab === 'audit' && (
                    <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                      
                      {/* Interactive Trajectory Plot */}
                      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <h3>
                              {selectedEpisodeId 
                                ? `Joint Kinematic Profile — Episode #${activeEpisodeForTrajectory.id}` 
                                : "Dataset Representative Kinematic Profile (Average)"}
                            </h3>
                            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                              {selectedEpisodeId 
                                ? `Displaying raw and smoothed angular velocities for curated Episode #${activeEpisodeForTrajectory.id}.`
                                : "Visualizes high-frequency noise and sudden corrections during teleoperation."}
                            </p>
                          </div>
                          
                          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px', background: 'rgba(255,255,255,0.03)', padding: '6px 12px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
                            <input 
                                type="checkbox" 
                                checked={applySmoothing} 
                                onChange={(e) => setApplySmoothing(e.target.checked)}
                                style={{ accentColor: 'var(--brand-gold)' }} 
                            />
                            <span>Trajectory Smoothing</span>
                          </label>
                        </div>

                        <div style={{ height: '300px', width: '100%', position: 'relative' }}>
                          <Line data={trajectoryChartData} options={trajectoryChartOptions} />
                        </div>
                      </div>

                      {/* Kinematics Metric Cards */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left', padding: '20px' }}>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>
                            {selectedEpisodeId ? `EPISODE #${activeEpisodeForTrajectory.id} JERK SPIKE` : "DATASET AVG JERK SPIKE"}
                          </span>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                            <span style={{ fontSize: '24px', fontWeight: 700 }}>
                              {activeEpisodeForTrajectory.jerkRate}%
                            </span>
                            <span className={activeEpisodeForTrajectory.jerkRate > 5.0 ? "status-badge status-badge-danger" : "status-badge status-badge-success"} style={{ fontSize: '10px', padding: '2px 6px' }}>
                              {activeEpisodeForTrajectory.jerkRate > 5.0 ? "FAIL > 5%" : "PASS < 5%"}
                            </span>
                          </div>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Outlier episodes containing abrupt acceleration spikes.</p>
                        </div>

                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left', padding: '20px' }}>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>
                            {selectedEpisodeId ? `EPISODE #${activeEpisodeForTrajectory.id} VEL DISCONTINUITY` : "DATASET AVG VEL DISCONTINUITY"}
                          </span>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                            <span style={{ fontSize: '24px', fontWeight: 700, color: activeEpisodeForTrajectory.velDiscontinuity > 4.0 ? 'var(--danger)' : 'var(--text-primary)' }}>
                              {activeEpisodeForTrajectory.velDiscontinuity}%
                            </span>
                            <span className={activeEpisodeForTrajectory.velDiscontinuity > 4.0 ? 'status-badge status-badge-danger' : 'status-badge status-badge-success'} style={{ fontSize: '10px', padding: '2px 6px' }}>
                              {activeEpisodeForTrajectory.velDiscontinuity > 4.0 ? "FAIL > 4%" : "PASS < 4%"}
                            </span>
                          </div>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Abrupt jumps indicating controller packet drops or lag.</p>
                        </div>

                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '6px', textAlign: 'left', padding: '20px' }}>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>
                            {selectedEpisodeId ? `EPISODE #${activeEpisodeForTrajectory.id} LDLJ SCORE` : "DATASET AVG LDLJ SCORE"}
                          </span>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                            <span style={{ fontSize: '24px', fontWeight: 700, color: (-4.0 - activeEpisodeForTrajectory.jerkRate * 0.8) < -10.0 ? 'var(--danger)' : 'var(--text-primary)' }}>
                              {(-4.0 - activeEpisodeForTrajectory.jerkRate * 0.8).toFixed(1)}
                            </span>
                            <span className={(-4.0 - activeEpisodeForTrajectory.jerkRate * 0.8) < -10.0 ? 'status-badge status-badge-danger' : 'status-badge status-badge-success'} style={{ fontSize: '10px', padding: '2px 6px' }}>
                              {(-4.0 - activeEpisodeForTrajectory.jerkRate * 0.8) < -10.0 ? "FAIL > -10" : "PASS > -10"}
                            </span>
                          </div>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Log-determinant of joint jerk. Lower means rougher path.</p>
                        </div>
                      </div>

                      {/* Anomaly & Flags List */}
                      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                        <h3 style={{ fontSize: '15px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <AlertTriangle size={16} className="text-amber-500" /> Audit Kinematic Anomaly Flags
                        </h3>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                          {activeFlags.map((flag, idx) => (
                            <div 
                              key={idx} 
                              style={{ 
                                border: '1px solid rgba(255,255,255,0.04)', 
                                background: 'rgba(255,255,255,0.01)', 
                                padding: '14px', borderRadius: '8px'
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                <span className={`status-badge ${flag.level === 'critical' ? 'status-badge-danger' : 'status-badge-warning'}`}>
                                  {flag.level}
                                </span>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>{flag.metric}</span>
                              </div>
                              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{flag.msg}</p>
                              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                                Observed: <strong style={{ color: '#fff' }}>{flag.observed}</strong> (Threshold: {flag.threshold})
                              </div>
                            </div>
                          ))}
                          
                          {activeFlags.length === 0 && (
                            <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '12px' }}>
                              No severe kinematic anomalies found. Trajectories are smooth.
                            </div>
                          )}
                        </div>
                      </div>

                    </div>
                  )}

                  {/* TAB 2: BEHAVIORAL CORESET SPACE (2D EMBEDDINGS) */}
                  {cliTab === 'coreset' && (
                    <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: '24px' }}>
                        
                        {/* Interactive Canvas Workspace */}
                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left', padding: '20px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                              <h3>Behavioral embedding projections</h3>
                              <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Each point represents a single episode. Gold circles represent the curated coreset.</p>
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>t-SNE DIMENSION REDUCTION</div>
                          </div>

                          <div style={{ display: 'flex', justifyContent: 'center', background: '#02050f', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.04)', position: 'relative', overflow: 'hidden' }}>
                            <canvas
                              ref={canvasRef}
                              width={480}
                              height={340}
                              onMouseMove={handleCanvasMouseMove}
                              onClick={handleCanvasClick}
                              onMouseLeave={() => setHoveredPoint(null)}
                              style={{ display: 'block', cursor: 'crosshair', maxWidth: '100%' }}
                            />
                            
                            {/* Legend labels */}
                            <div style={{ position: 'absolute', bottom: '12px', left: '12px', display: 'flex', gap: '12px', fontSize: '11px', background: 'rgba(4, 8, 20, 0.85)', padding: '6px 12px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--brand-gold)' }} />
                                <span>Coreset</span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'rgba(148, 163, 184, 0.3)' }} />
                                <span>Pruned</span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span style={{ color: 'var(--danger)', fontWeight: 700 }}>×</span>
                                <span>Anomaly</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Coreset summary sidebar */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '14px', padding: '20px', textAlign: 'left' }}>
                            <h3>Coreset Curation Results</h3>
                            
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '12px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Selected Coreset:</span>
                                <strong style={{ color: 'var(--brand-gold)' }}>{curationResult.totalKept} episodes</strong>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Redundancy Pruned:</span>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{curationResult.totalRedundant}</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Rejected Anomalies:</span>
                                <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{curationResult.totalRejected}</span>
                              </div>
                            </div>

                            <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                              Using Stage 1 Quality filtering + Farthest Point state-space selection to cover full robot work envelopes.
                            </div>
                          </div>

                           {/* Selected / Hovered Episode details */}
                           <div className="glass-card animate-fade-in-up" style={{ padding: '20px', textAlign: 'left', minHeight: '160px', display: 'flex', flexDirection: 'column' }}>
                             <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                               <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                                 {hoveredPoint ? "HOVERING EPISODE INFO" : selectedEpisodeId ? "LOCKED EPISODE INFO" : "EPISODE INFO"}
                               </span>
                               {selectedEpisodeId && !hoveredPoint && (
                                 <button 
                                   onClick={() => setSelectedEpisodeId(null)} 
                                   style={{ background: 'transparent', border: 'none', color: 'var(--brand-gold)', fontSize: '10px', fontWeight: 600, cursor: 'pointer', textTransform: 'uppercase', padding: 0 }}
                                 >
                                   Reset Lock
                                 </button>
                               )}
                             </div>
                             
                             {hoveredPoint || (selectedEpisodeId && embeddingPoints.find(p => p.id === selectedEpisodeId)) ? (
                               (() => {
                                 const ep = hoveredPoint || embeddingPoints.find(p => p.id === selectedEpisodeId);
                                 return (
                                   <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                     <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                       <strong style={{ fontSize: '15px' }}>Episode #{ep.id} {selectedEpisodeId === ep.id && !hoveredPoint && "🔒"}</strong>
                                       <span className={`status-badge ${ep.isQualityAnomaly ? 'status-badge-danger' : 'status-badge-success'}`} style={{ fontSize: '9px', padding: '1px 6px' }}>
                                         {ep.isQualityAnomaly ? 'FAIL' : 'PASS'}
                                       </span>
                                     </div>
                                     <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                                       Segment: <em>{ep.clusterName}</em>
                                     </div>
                                     <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                       <span>Jerk Spike: <strong>{ep.jerkRate}%</strong></span>
                                       <span>Discontinuity: <strong>{ep.velDiscontinuity}%</strong></span>
                                     </div>
                                     <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                       Action space Entropy: <strong>{ep.entropy} bits</strong>
                                     </div>
                                   </div>
                                 );
                               })()
                             ) : (
                               <div style={{ color: 'var(--text-muted)', fontSize: '12.5px', marginTop: 'auto', marginBottom: 'auto', fontStyle: 'italic', textAlign: 'center' }}>
                                 Hover cursor or click on embedding dots to trace telemetry properties.
                               </div>
                             )}
                           </div>

                        </div>

                      </div>

                    </div>
                  )}

                  {/* TAB 3: REFERENCE BENCHMARK COMPARE */}
                  {cliTab === 'compare' && (
                    <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                      
                      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px', textAlign: 'left' }}>
                        <h3>Benchmark Delta Report (vs. target reference task)</h3>
                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Programmatic comparison of telemetry distributions. Red indicates significantly worse variances than model benchmark parameters.</p>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '10px' }}>
                          
                          {/* Row 1: Vel Discontinuity */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13.5px' }}>
                              <span>Velocity Discontinuity Rate (Yours vs. Reference)</span>
                              <strong style={{ color: datasetMetrics.velDiscontinuity > 4.0 ? 'var(--danger)' : 'var(--success)' }}>
                                {datasetMetrics.velDiscontinuity}% vs. {ROBOT_DATASETS.aloha_mobile.velDiscontinuity}% (Delta: {((datasetMetrics.velDiscontinuity - ROBOT_DATASETS.aloha_mobile.velDiscontinuity) >= 0 ? "+" : "") + (datasetMetrics.velDiscontinuity - ROBOT_DATASETS.aloha_mobile.velDiscontinuity).toFixed(1)}%)
                              </strong>
                            </div>
                            <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden', display: 'flex' }}>
                              <div style={{ width: `${Math.min(100, Math.max(5, (datasetMetrics.velDiscontinuity / 25) * 100))}%`, background: datasetMetrics.velDiscontinuity > 4.0 ? 'var(--danger)' : 'var(--brand-gold)' }} />
                            </div>
                          </div>

                          {/* Row 2: Jerk Spike Rate */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13.5px' }}>
                              <span>Jerk Spike Rate (Yours vs. Reference)</span>
                              <strong style={{ color: datasetMetrics.jerkRate > 5.0 ? 'var(--danger)' : 'var(--success)' }}>
                                {datasetMetrics.jerkRate}% vs. {ROBOT_DATASETS.aloha_mobile.jerkSpikeRate}% (Delta: {((datasetMetrics.jerkRate - ROBOT_DATASETS.aloha_mobile.jerkSpikeRate) >= 0 ? "+" : "") + (datasetMetrics.jerkRate - ROBOT_DATASETS.aloha_mobile.jerkSpikeRate).toFixed(1)}%)
                              </strong>
                            </div>
                            <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden', display: 'flex' }}>
                              <div style={{ width: `${Math.min(100, Math.max(5, (datasetMetrics.jerkRate / 15) * 100))}%`, background: datasetMetrics.jerkRate > 5.0 ? 'var(--danger)' : 'var(--brand-gold)' }} />
                            </div>
                          </div>

                          {/* Row 3: Mean LDLJ path smoothness */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13.5px' }}>
                              <span>Mean LDLJ Path Smoothness (Yours vs. Reference)</span>
                              <strong style={{ color: datasetMetrics.ldlj < -10.0 ? 'var(--danger)' : 'var(--success)' }}>
                                {datasetMetrics.ldlj} vs. {ROBOT_DATASETS.aloha_mobile.ldlj} (Delta: {((datasetMetrics.ldlj - ROBOT_DATASETS.aloha_mobile.ldlj) >= 0 ? "+" : "") + (datasetMetrics.ldlj - ROBOT_DATASETS.aloha_mobile.ldlj).toFixed(1)})
                              </strong>
                            </div>
                            <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden', display: 'flex' }}>
                              <div style={{ width: `${Math.min(100, Math.max(5, (Math.abs(datasetMetrics.ldlj) / 20) * 100))}%`, background: datasetMetrics.ldlj < -10.0 ? 'var(--danger)' : 'var(--brand-gold)' }} />
                            </div>
                          </div>

                        </div>
                      </div>

                      {/* Actionable recommendation list */}
                      <div className="glass-card animate-fade-in-up" style={{ textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <h3>Recommended Quality Adjustments</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                          <div style={{ borderLeft: '3px solid var(--brand-gold)', paddingLeft: '14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            <span style={{ fontSize: '13.5px', fontWeight: 600 }}>Filter Kinematic Outliers:</span>
                            <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Pruning the 9 worst jerk-spiking episodes improves joint smoothness variance by 58.8%.</span>
                          </div>
                          <div style={{ borderLeft: '3px solid #0ea5e9', paddingLeft: '14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            <span style={{ fontSize: '13.5px', fontWeight: 600 }}>Apply Trajectory Action Resampling:</span>
                            <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>Applying Savitzky-Golay filtering decreases velocity discontinuity to 2.1%, allowing stable policy execution.</span>
                          </div>
                        </div>
                      </div>

                    </div>
                  )}

                  {/* TAB 4: REAL-TIME WATCH SIMULATION */}
                  {cliTab === 'watch' && (
                    <div className="animate-fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
                        
                        {/* Live chart simulator */}
                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <div className="pulse-dot" />
                              <h3>Live Telemetry Stream: calibra watch</h3>
                            </div>
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>15Hz CONTROL LOOP</span>
                          </div>
                          
                          <div style={{ padding: '24px', background: '#02050f', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.04)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', height: '220px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>STREAMING JOINT VELOCITIES</span>
                            <div style={{ fontSize: '32px', fontWeight: 800, color: 'var(--success)', fontFamily: 'var(--font-mono)' }}>
                              {streamVelocity.toFixed(4)} rad/s
                            </div>
                            <div style={{ display: 'flex', gap: '3px', alignItems: 'flex-end', height: '60px', width: '100%' }}>
                              {Array.from({ length: 48 }).map((_, i) => (
                                <div 
                                  key={i} 
                                  style={{ 
                                    flex: 1, 
                                    height: `${Math.max(10, Math.sin((Date.now() + i * 200) / 1000) * 20 + 35)}%`, 
                                    background: 'var(--success)',
                                    borderRadius: '2px',
                                    opacity: 0.7 
                                  }} 
                                />
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Stream Audit logs */}
                        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '12px', textAlign: 'left' }}>
                          <h3>Watch Operator Feedback</h3>
                          
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '220px', overflowY: 'auto', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                            {watchLog.map((log, idx) => (
                              <div 
                                key={idx} 
                                style={{ 
                                  display: 'flex', 
                                  gap: '8px', 
                                  color: log.msg.includes('Gripper') || log.msg.includes('lag') ? 'var(--warning)' : 'var(--text-secondary)',
                                  borderBottom: '1px solid rgba(255,255,255,0.02)',
                                  paddingBottom: '8px'
                                }}
                              >
                                <span style={{ color: 'var(--text-muted)' }}>[{log.time}]</span>
                                <span>{log.msg}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                      </div>

                    </div>
                  )}

                </div>

              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer Banner */}
      <footer style={{ borderTop: '1px solid var(--border-color)', position: 'absolute', bottom: 0, left: 0, right: 0, height: '70px', display: 'flex', alignItems: 'center', background: 'rgba(4,8,21,0.5)', backdropFilter: 'blur(8px)' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12.5px', color: 'var(--text-muted)' }}>
          <span>© 2026 Calibra Robotics. Released under the MIT License.</span>
          <div style={{ display: 'flex', gap: '20px' }}>
            <a href="https://github.com/omerTT/Calibra" target="_blank" rel="noreferrer" style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Code size={14} /> GitHub Repository
            </a>
            <span>Observability & Coreset Curation for Robotics</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
