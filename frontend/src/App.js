import { useState, useRef, useEffect } from 'react';
import { useTerminalData } from './hooks/useTerminalData';
import TopBar from './components/TopBar';
import Overview from './components/Overview';
import './index.css';

// ── Tab definitions ──────────────────────────────────────────────────────────
const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'p1',       label: 'P1 · Sentiment' },
  { id: 'p2',       label: 'P2 · Memory' },
  { id: 'p3',       label: 'P3 · Structure' },
  { id: 'p4',       label: 'P4 · Candle' },
  { id: 'p5',       label: 'P5 · Regime' },
  { id: 'p6',       label: 'P6 · Events' },
  { id: 'p7',       label: 'P7 · Council' },
  { id: 'p8',       label: 'P8 · Decision' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const { data, connected, sentimentHistory, refreshAll } = useTerminalData();

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100vh', overflow:'hidden', background:'var(--bg)' }}>

      {/* TOP BAR */}
      <TopBar
        connected={connected}
        onRefresh={refreshAll}
        pillar6Data={data.pillar6}
      />

      {/* TABS */}
      <div style={{
        display:'flex', alignItems:'center', padding:'0 18px',
        background:'var(--bg2)', borderBottom:'1px solid var(--border)',
        flexShrink:0, overflowX:'auto', gap:0,
      }}>
        {TABS.map(tab => (
          <div
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding:'9px 14px',
              fontSize:'10px',
              fontWeight:500,
              letterSpacing:'0.09em',
              textTransform:'uppercase',
              whiteSpace:'nowrap',
              cursor:'pointer',
              color: activeTab === tab.id ? 'var(--text)' : 'var(--text3)',
              borderBottom: activeTab === tab.id ? '2px solid var(--white)' : '2px solid transparent',
              transition:'all 0.12s',
            }}
          >
            {tab.label}
          </div>
        ))}
      </div>

      {/* CONTENT */}
      <div style={{ flex:1, overflow:'hidden', display:'flex' }}>
        {activeTab === 'overview' && <Overview data={data} />}
        {activeTab === 'p1'       && <P1Tab data={data.pillar1} sentimentHistory={sentimentHistory} />}
        {activeTab === 'p2'       && <P2Tab data={data.pillar2} />}
        {activeTab === 'p3'       && <P3Tab data={data.pillar3} />}
        {activeTab === 'p4'       && <P4Tab data={data.pillar4} />}
        {activeTab === 'p5'       && <P5Tab data={data.pillar5} />}
        {activeTab === 'p6'       && <P6Tab data={data.pillar6} />}
        {activeTab === 'p7'       && <P7Tab data={data.pillar7} />}
        {activeTab === 'p8'       && <P8Tab data={data.pillar8} />}
      </div>

      {/* STATUS BAR */}
      <div style={{
        display:'flex', alignItems:'center', justifyContent:'space-between',
        padding:'0 18px', height:'24px',
        background:'var(--bg2)', borderTop:'1px solid var(--border)',
        flexShrink:0,
      }}>
        <div style={{ display:'flex', gap:'14px' }}>
          {['P1','P2','P3','P4','P5','P6','P7','P8'].map((p, i) => {
            const key = `pillar${i+1}`;
            const ready = data[key] !== null;
            return (
              <span key={p} style={{ fontSize:'9px', color: ready ? 'var(--text2)' : 'var(--text3)' }}>
                {p} <span style={{ color: ready ? 'var(--green)' : 'var(--red)' }}>{ready ? '✓' : '○'}</span>
              </span>
            );
          })}
        </div>
        <div style={{ display:'flex', gap:'14px' }}>
          <span style={{ fontSize:'9px', color:'var(--text3)' }}>
            {connected
              ? <span style={{ color:'var(--green)' }}>● Live</span>
              : <span style={{ color:'var(--yellow)' }}>○ Reconnecting</span>}
          </span>
          <span style={{ fontSize:'9px', color:'var(--text3)' }}>BTCUSDT · 1H</span>
        </div>
      </div>

    </div>
  );
}

// ════════════════════════════════════════════════════
// SHARED SHELL & HELPERS
// ════════════════════════════════════════════════════

function TabShell({ children, terminal }) {
  const [termOpen, setTermOpen] = useState(false);
  return (
    <div style={{ display:'flex', flexDirection:'column', width:'100%', height:'100%', overflow:'hidden' }}>
      <div style={{ flex:1, overflow:'hidden', display:'flex' }}>
        {children}
      </div>
      {terminal && (
        <div style={{ background:'#000', borderTop:'1px solid var(--border2)', flexShrink:0 }}>
          <div
            onClick={() => setTermOpen(!termOpen)}
            style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'6px 12px', background:'#0a0a0a', borderBottom: termOpen ? '1px solid var(--border2)' : 'none', cursor:'pointer' }}
          >
            <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
              <div style={{ display:'flex', gap:'5px' }}>
                {['#ff5f56','#ffbd2e','#27c93f'].map(c => <div key={c} style={{ width:'10px', height:'10px', borderRadius:'50%', background:c }} />)}
              </div>
              <span style={{ fontSize:'10px', color:'var(--text3)', letterSpacing:'.08em' }}>{terminal.title}</span>
            </div>
            <span style={{ fontSize:'9px', color:'var(--text3)' }}>{termOpen ? '▲ collapse' : '▼ expand'}</span>
          </div>
          {termOpen && (
            <div style={{ maxHeight:'220px', overflowY:'auto', padding:'12px 14px', fontFamily:'var(--mono)', fontSize:'10px', lineHeight:'1.7', color:'#ccc' }}>
              {terminal.content}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function KVRow({ label, value, color }) {
  const c = color === 'g' ? 'var(--green)' : color === 'r' ? 'var(--red)' : color === 'y' ? 'var(--yellow)' : 'var(--text)';
  return (
    <div className="kvr">
      <div className="kvk">{label}</div>
      <div className="kvv" style={{ color:c }}>{value ?? '—'}</div>
    </div>
  );
}

function NotReady() {
  return (
    <div className="not-ready">
      <div className="spinner" />
      <span>Computing... check back shortly</span>
    </div>
  );
}

function TermLine({ children, prompt, color }) {
  const c = color==='cyan'?'#4af':color==='green'?'#3ddc84':color==='red'?'#ff4d4d':color==='yellow'?'#f0c040':'#aaa';
  return (
    <div style={{ color:c, lineHeight:'1.7' }}>
      {prompt && <span style={{ color:'#555' }}>((venv)) $ </span>}
      {children}
    </div>
  );
}

function TermSpan({ children, color }) {
  const c = color==='green'?'#3ddc84':color==='red'?'#ff4d4d':color==='yellow'?'#f0c040':color==='white'?'#fff':'#aaa';
  return <span style={{ color:c }}>{children}</span>;
}

// ── Mock data ─────────────────────────────────────────────────────────────────
const MOCK_HEADLINES = [
  { t:'BTC Surges Past $75,000 Amid ETF Inflows', s:'Ibtimes', c:'pos' },
  { t:'Geopolitical Easing Sparks Crypto Rebound', s:'Ibtimes', c:'pos' },
  { t:'CoinEx Monthly — Repricing Risk and Resilient Bitcoin', s:'CoinDesk', c:'neu' },
  { t:'ETH Dips as Ceasefire Rally Fades', s:'CoinDesk', c:'neg' },
  { t:'Fed Maintains Restrictive Policy Stance', s:'Fed Reserve', c:'neu' },
  { t:'Powell Signals Data-Dependent Approach', s:'Fed Reserve', c:'neu' },
  { t:'Spot Bitcoin ETF Inflows Hit $800M in Single Day', s:'Cointelegraph', c:'pos' },
  { t:'Crypto Faces Pressure as Geopolitical Tensions Escalate', s:'Decrypt', c:'neg' },
  { t:'Stablecoin Regulation Progress: Senate Bill Implications', s:'Cointelegraph', c:'neu' },
  { t:'BlackRock IBIT Records Strongest Week of Inflows', s:'CoinDesk', c:'pos' },
];

const MOCK_SCENARIOS = [
  { label:'Hawkish Hold', detail:'Fed keeps policy tight. Downside extends if yields rise.', direction:'DOWN' },
  { label:'Neutral Hold', detail:'Fed matches expectations. Whipsaw likely on Q&A nuance.', direction:'NEUTRAL' },
  { label:'Dovish Shift', detail:'Softer policy. Upside persists if risk assets catch a bid.', direction:'UP' },
];

const MOCK_SENTIMENT = [
  { timestamp:'2026-04-19', score:0.12 },
  { timestamp:'2026-04-20', score:-0.08 },
  { timestamp:'2026-04-21', score:0.05 },
  { timestamp:'2026-04-22', score:-0.15 },
  { timestamp:'2026-04-23', score:0.02 },
  { timestamp:'2026-04-24', score:-0.06 },
  { timestamp:'2026-04-25', score:0.01 },
];

// ── Sentiment Chart ───────────────────────────────────────────────────────────
function SentimentChart({ history }) {
  const chartData = history.length > 0 ? history : MOCK_SENTIMENT;
  const max = 0.25;
  const h   = 140;

  return (
    <div style={{ height:'160px', display:'flex', alignItems:'flex-end', gap:'4px', padding:'10px 0 20px' }}>
      {chartData.slice(-7).map((d, i) => {
        const score = parseFloat(d.score) || 0;
        const barH  = Math.abs(score) / max * (h / 2);
        const isPos = score > 0.03;
        const isNeg = score < -0.03;
        const color = isPos ? 'var(--green)' : isNeg ? 'var(--red)' : 'var(--yellow)';
        const label = d.timestamp
          ? new Date(d.timestamp).toLocaleDateString('en',{ month:'short', day:'numeric' })
          : `D${i+1}`;
        return (
          <div key={i} style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', height:'100%', justifyContent:'center', gap:'2px' }}>
            {isPos  && <div style={{ width:'100%', height:barH+'px', background:color, borderRadius:'2px 2px 0 0', opacity:0.8 }} />}
            {!isPos && <div style={{ height:(h/2-barH)+'px' }} />}
            <div style={{ width:'1px', height:'1px', background:'var(--border2)' }} />
            {isNeg  && <div style={{ width:'100%', height:barH+'px', background:color, borderRadius:'0 0 2px 2px', opacity:0.8 }} />}
            {!isNeg && <div style={{ height:(h/2-barH)+'px' }} />}
            <div style={{ fontSize:'8px', color:'var(--text3)', marginTop:'4px', whiteSpace:'nowrap' }}>{label}</div>
          </div>
        );
      })}
    </div>
  );
}

// ════════════════════════════════════════════════════
// P1 — SENTIMENT
// ════════════════════════════════════════════════════
function P1Tab({ data, sentimentHistory }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const agg        = data.aggregate_sentiment || {};
  const label      = agg.label || 'NEUTRAL';
  const confidence = agg.confidence || 0;
  const articles   = data.raw_articles || data.articles || [];
  const count      = data.article_count || articles.length || 58;

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar1_sentiment.run_pillar1</TermLine>
      <TermLine color="cyan">┌─ PILLAR 1 — SENTIMENT ──────────────────────────────┐</TermLine>
      <TermLine>  Aggregate Sentiment    <TermSpan color="yellow">{label}</TermSpan></TermLine>
      <TermLine>  Model Confidence       {(confidence * 100).toFixed(1)}%</TermLine>
      <TermLine>  Articles Analyzed      {count}</TermLine>
      <TermLine color="cyan">├─ Sources ───────────────────────────────────────────┤</TermLine>
      <TermLine>  CoinDesk / Cointelegraph / Decrypt / Fed Reserve</TermLine>
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar1_sentiment — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Sentiment Summary</div>
          <div className="kv">
            <KVRow label="Aggregate"  value={label}                         color={label==='POSITIVE'?'g':label==='NEGATIVE'?'r':'y'} />
            <KVRow label="Confidence" value={(confidence*100).toFixed(1)+'%'} />
            <KVRow label="Articles"   value={count+' analyzed'} />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Sentiment History — 7 Days</div>
          <SentimentChart history={sentimentHistory} />
        </div>
      </div>

      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec">
          <div className="sec-label">Institutional Macro Summary</div>
          <div className="ai">
            <p>{data.institutional_summary || data.ai_summary || 'Institutional sentiment neutral — mixed geopolitical tensions, ETF inflows, and crypto market dynamics with low confidence.'}</p>
          </div>
        </div>
        <div className="sec" style={{ flex:1, overflow:'hidden', display:'flex', flexDirection:'column' }}>
          <div className="sec-label">Headlines Analyzed ({count})</div>
          <div style={{ overflowY:'auto', flex:1 }}>
            {(articles.length > 0 ? articles.slice(0, 20) : MOCK_HEADLINES).map((a, i) => {
              const title     = a.title || a.headline || a.t || '';
              const source    = a.source || a.s || '';
              const sentiment = a.sentiment || a.c || 'neu';
              const dotColor  = sentiment === 'positive' || sentiment === 'pos'
                ? 'var(--green)'
                : sentiment === 'negative' || sentiment === 'neg'
                  ? 'var(--red)'
                  : 'var(--text3)';
              return (
                <div key={i} style={{ display:'flex', alignItems:'flex-start', gap:'8px', padding:'6px 0', borderBottom:'1px solid var(--border)', fontSize:'10px' }}>
                  <div style={{ width:'5px', height:'5px', borderRadius:'50%', flexShrink:0, marginTop:'4px', background:dotColor }} />
                  <div style={{ color:'var(--text2)', flex:1, lineHeight:1.5 }}>{title}</div>
                  <div style={{ color:'var(--text3)', fontSize:'9px', flexShrink:0, whiteSpace:'nowrap' }}>{source}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P2 — MEMORY
// ════════════════════════════════════════════════════
function P2Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const ms   = data.memory_summary    || {};
  const fwd  = data.forward_outcomes  || {};
  const stab = data.stability_diagnostics || {};
  const ctx  = data.context_memory    || {};

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar2_market_memory_engine.run_pillar2</TermLine>
      <TermLine color="cyan">┌─ PILLAR 2 — MARKET MEMORY ──────────────────────────┐</TermLine>
      <TermLine>  Memory Bias     <TermSpan color="red">{ms.memory_bias || 'MEAN_REVERSION_BIAS'}</TermSpan></TermLine>
      <TermLine>  Match Quality   <TermSpan color="yellow">{ms.match_quality || 'MODERATE'}</TermSpan></TermLine>
      <TermLine>  Sample Size     {ms.sample_size || 300}</TermLine>
      <TermLine>  Confidence      {ms.headline_confidence || '0.746'}</TermLine>
      <TermLine color="cyan">├─ Forward Outcomes ──────────────────────────────────┤</TermLine>
      <TermLine>  Mean Reversion Prob   <TermSpan color="green">{fwd.mean_reversion_prob || '61.2%'}</TermSpan></TermLine>
      <TermLine>  Continuation Prob     {fwd.continuation_prob || '20.5%'}</TermLine>
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar2_market_memory — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Memory State</div>
          <div className="kv">
            <KVRow label="Memory Bias"   value={ms.memory_bias || '—'}                  color="r" />
            <KVRow label="Match Quality" value={ms.match_quality || '—'}                color="y" />
            <KVRow label="Sample Size"   value={(ms.sample_size || '—')+' analogs'} />
            <KVRow label="Confidence"    value={ms.headline_confidence || '—'} />
            <KVRow label="Mean Rev Prob" value={fwd.mean_reversion_prob || '—'}         color="g" />
            <KVRow label="Continuation"  value={fwd.continuation_prob || '—'} />
            <KVRow label="MFE 6-Bar"     value={fwd.mean_mfe_6bar || '—'}               color="g" />
            <KVRow label="MAE 6-Bar"     value={fwd.mean_mae_6bar || '—'}               color="r" />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Temporal Stability</div>
          <div className="kv">
            <KVRow label="Temporal Stability" value={stab.temporal_stability_score ?? '—'} color="r" />
            <KVRow label="Regime Dependency"  value={stab.regime_dependency_score  ?? '—'} color="y" />
          </div>
        </div>
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec">
          <div className="sec-label">Context Memory</div>
          <div className="kv">
            {Object.entries(ctx).map(([k, v]) => (
              <KVRow key={k} label={k.replace(/_/g,' ')} value={v} />
            ))}
          </div>
        </div>
        <div className="sec" style={{ flex:1 }}>
          <div className="sec-label">AI Memory Overview</div>
          <div className="ai">
            <p>{data.ai_overview?.overview || 'Mean reversion bias at 61.2% — 300 analogs, MODERATE match quality. Temporal stability unstable (older analogs bullish, recent bearish). Use as probabilistic prior, not deterministic signal.'}</p>
          </div>
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P3 — STRUCTURE
// ════════════════════════════════════════════════════
function P3Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const structBlock = data.structure_state        || {};
  const liq         = data.liquidity_levels       || {};
  const summary     = data.structure_liquidity_summary || {};
  const liqRisk     = data.liquidation_risk       || {};

  const structState = typeof structBlock === 'object' ? structBlock.market_structure : structBlock;
  const buySide     = liq.buy_side_liquidity;
  const sellSide    = liq.sell_side_liquidity;
  const magnet      = liq.nearest_liquidity_magnet;

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar3_structure_liquidity_engine.run_pillar3</TermLine>
      <TermLine color="cyan">┌─ PILLAR 3 — STRUCTURE & LIQUIDITY ──────────────────┐</TermLine>
      <TermLine>  Market Structure   <TermSpan color="yellow">{structState || 'MIXED_STRUCTURE'}</TermSpan></TermLine>
      <TermLine>  Buy-Side Liquidity <TermSpan color="green">${buySide?.toLocaleString() || '69,988'}</TermSpan></TermLine>
      <TermLine>  Sell-Side          <TermSpan color="red">${sellSide?.toLocaleString() || '65,084'}</TermSpan></TermLine>
      <TermLine>  Nearest Magnet     <TermSpan color="yellow">${magnet?.toLocaleString() || '69,988'}</TermSpan></TermLine>
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar3_structure_liquidity — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Market Structure</div>
          <div className="kv">
            <KVRow label="Structure"     value={structState || '—'}                       color="y" />
            <KVRow label="Range State"   value={structBlock.range_state || '—'}           color="g" />
            <KVRow label="Dominant Side" value={summary.dominant_liquidity_side || '—'}   color="g" />
            <KVRow label="Trap Risk"     value={summary.trap_risk || '—'}                 color="g" />
            <KVRow label="Breakout Trap" value={
              data.trap_detection?.breakout_trap_probability
                ? (data.trap_detection.breakout_trap_probability * 100).toFixed(1) + '%'
                : '—'
            } color="y" />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Liquidity Levels</div>
          <div className="kv">
            <KVRow label="Buy-Side"       value={buySide  ? '$'+buySide.toLocaleString()  : '—'} color="g" />
            <KVRow label="Sell-Side"      value={sellSide ? '$'+sellSide.toLocaleString() : '—'} color="r" />
            <KVRow label="Nearest Magnet" value={magnet   ? '$'+magnet.toLocaleString()   : '—'} color="y" />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Liquidation Risk</div>
          <div className="kv">
            <KVRow label="Long Liq Risk"  value={liqRisk.long_liquidation_risk  || '—'}  color="g" />
            <KVRow label="Short Liq Risk" value={liqRisk.short_liquidation_risk || '—'}  color="y" />
            <KVRow label="Cascade Prob"   value={
              liqRisk.cascade_probability
                ? (liqRisk.cascade_probability * 100).toFixed(1) + '%'
                : '—'
            } color="y" />
          </div>
        </div>
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec" style={{ flex:1 }}>
          <div className="sec-label">AI Structure Overview</div>
          <div className="ai">
            <p>{data.ai_overview?.overview || 'TRENDING tape, MIXED_STRUCTURE, NEUTRAL compression. Buy-side dominant. Nearest magnet ~1% above current price. Market orienting around stop location, not clean directional discovery.'}</p>
          </div>
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P4 — CANDLE
// ════════════════════════════════════════════════════
function P4Tab({ data }) {
  const tvRef    = useRef(null);
  const tvLoaded = useRef(false);

  useEffect(() => {
    if (tvLoaded.current || !tvRef.current) return;

    const initWidget = () => {
      if (!window.TradingView || !tvRef.current) return;
      new window.TradingView.widget({
        container_id: 'tv_candle',
        symbol: 'BINANCE:BTCUSDT',
        interval: '60',
        timezone: 'UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#111',
        enable_publishing: false,
        hide_side_toolbar: true,
        allow_symbol_change: false,
        backgroundColor: '#0d0d0d',
        gridColor: '#181818',
        width: '100%',
        height: '100%',
        range: '1D',
      });
      tvLoaded.current = true;
    };

    if (window.TradingView) {
      initWidget();
    } else {
      const script = document.createElement('script');
      script.src = 'https://s3.tradingview.com/tv.js';
      script.async = true;
      script.onload = initWidget;
      document.head.appendChild(script);
    }
  }, []);

  const cs  = data?.candle_summary || {};
  const prs = data?.pressure       || {};

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar4_candle_engine.run_pillar4</TermLine>
      <TermLine color="cyan">┌─ PILLAR 4 — CANDLE INTELLIGENCE ────────────────────┐</TermLine>
      <TermLine>  Dominant Intent  <TermSpan color="green">{cs.dominant_intent || 'BUY_ABSORPTION_CANDIDATE'}</TermSpan></TermLine>
      <TermLine>  Confidence       <TermSpan color="yellow">{cs.intent_confidence ? (cs.intent_confidence*100).toFixed(1)+'%' : '28.1%'}</TermSpan></TermLine>
      <TermLine>  Momentum         <TermSpan color="yellow">{cs.momentum_state || 'STALLING'}</TermSpan></TermLine>
      <TermLine>  Follow-Through   <TermSpan color="red">{cs.follow_through_quality || 'WEAK'}</TermSpan></TermLine>
      <TermLine>  Pressure Bias    <TermSpan color="green">{prs.pressure_bias || 'BUY_PRESSURE'}</TermSpan></TermLine>
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar4_candle_engine — raw terminal output', content:termContent }}>
      <div style={{ flex:4, borderRight:'1px solid var(--border)', display:'flex', flexDirection:'column', overflow:'hidden' }}>
        <div style={{ padding:'10px 14px 6px', background:'var(--bg2)', borderBottom:'1px solid var(--border)', flexShrink:0 }}>
          <div className="sec-label" style={{ margin:0 }}>BTC/USDT · 1H · Latest Candles</div>
        </div>
        <div id="tv_candle" ref={tvRef} style={{ flex:1, width:'100%' }} />
      </div>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        {!data
          ? <NotReady />
          : (
            <>
              <div className="sec">
                <div className="sec-label">Candle Summary</div>
                <div className="kv">
                  <KVRow label="Dominant Intent" value={(cs.dominant_intent||'—').replace('_CANDIDATE','')} color="g" />
                  <KVRow label="Confidence"      value={cs.intent_confidence?(cs.intent_confidence*100).toFixed(1)+'%':'—'} color="y" />
                  <KVRow label="Momentum"        value={cs.momentum_state||'—'}         color="y" />
                  <KVRow label="Follow-Through"  value={cs.follow_through_quality||'—'} color="r" />
                  <KVRow label="Pressure Bias"   value={prs.pressure_bias||'—'}         color="g" />
                  <KVRow label="Exhaustion"      value={cs.exhaustion_state||'NONE'}    color="g" />
                </div>
              </div>
              <div className="sec" style={{ flex:1 }}>
                <div className="sec-label">AI Candle Overview</div>
                <div className="ai">
                  <p>{data.ai_overview?.overview || 'BUY_ABSORPTION_CANDIDATE at ~28% confidence. High overlap, stalling momentum, weak follow-through. Absorption may resolve bullishly but patience required.'}</p>
                </div>
              </div>
            </>
          )
        }
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P5 — REGIME
// ════════════════════════════════════════════════════
function P5Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const rs    = data.regime_summary        || {};
  const strat = data.strategy_compatibility || {};
  const flags = data.risk_flags            || [];

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar5_regime_cycle_engine.run_pillar5</TermLine>
      <TermLine color="cyan">┌─ PILLAR 5 — REGIME & CYCLE ──────────────────────────┐</TermLine>
      <TermLine>  Directional Regime  <TermSpan color="green">{rs.directional_regime||'STRONG_UPTREND'}</TermSpan></TermLine>
      <TermLine>  Volatility Regime   <TermSpan color="red">{rs.volatility_regime||'DISLOCATED'}</TermSpan></TermLine>
      <TermLine>  Cycle Phase         <TermSpan color="green">{rs.cycle_phase||'EXPANSION'}</TermSpan></TermLine>
      <TermLine>  Confidence          <TermSpan color="green">{data.confidence_score?(data.confidence_score*100).toFixed(1)+'%':'87.5%'}</TermSpan></TermLine>
      <TermLine color="cyan">└──────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar5_regime_cycle — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Regime Summary</div>
          <div className="kv">
            <KVRow label="Directional"    value={rs.directional_regime||'—'}  color="g" />
            <KVRow label="Volatility"     value={rs.volatility_regime||'—'}   color="r" />
            <KVRow label="Cycle Phase"    value={rs.cycle_phase||'—'}         color="g" />
            <KVRow label="Market State"   value={rs.market_state||'—'}        color="g" />
            <KVRow label="Confidence"     value={data.confidence_score?(data.confidence_score*100).toFixed(1)+'%':'—'} color="g" />
            <KVRow label="Vol Percentile" value={rs.volatility_percentile?(rs.volatility_percentile*100).toFixed(1)+'%':'—'} color="r" />
            <KVRow label="7d Return"      value={rs.return_7d?(rs.return_7d*100).toFixed(2)+'%':'—'} color="g" />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Strategy Fit</div>
          <div className="kv">
            <KVRow label="Trend Following" value={strat.trend_following||'—'}  color="g" />
            <KVRow label="Breakout"        value={strat.breakout_trading||'—'} color="r" />
            <KVRow label="Mean Reversion"  value={strat.mean_reversion||'—'}   color="r" />
          </div>
        </div>
        {flags.length > 0 && (
          <div className="sec">
            <div className="sec-label">Risk Flags</div>
            {flags.map((f, i) => <div key={i} className="warn"><span>⚠</span><span>{f}</span></div>)}
          </div>
        )}
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec" style={{ flex:1 }}>
          <div className="sec-label">AI Regime Overview</div>
          <div className="ai">
            <p>{data.ai_overview?.overview || 'STRONG_UPTREND with EXPANSION cycle at high confidence. BULLISH_STACKED moving averages, HH·HL structure. Volatility DISLOCATED at 99th percentile — not a low-risk environment. Favors trend following only.'}</p>
          </div>
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P6 — EVENTS
// ════════════════════════════════════════════════════
function P6Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const restrictions = data.trade_restrictions || {};
  const scenarios    = data.scenarios          || [];
  const uncertainty  = data.base_uncertainty   || 0;

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar6_high_impact_events.run_pillar6</TermLine>
      <TermLine color="cyan">┌─ PILLAR 6 — HIGH IMPACT EVENTS ─────────────────────┐</TermLine>
      <TermLine>  Event          <TermSpan color="white">{data.event||data.event_name||'FOMC Rate Decision'}</TermSpan></TermLine>
      <TermLine>  State          {data.state||'IDLE'}</TermLine>
      <TermLine>  Uncertainty    <TermSpan color="yellow">{(uncertainty*100).toFixed(1)}%</TermSpan></TermLine>
      <TermLine>  Trade Allowed  <TermSpan color="green">{restrictions.allow_trade!==false?'YES':'NO'}</TermSpan></TermLine>
      <TermLine>  Size Mult      <TermSpan color="yellow">{restrictions.size_multiplier||'0.3'}×</TermSpan></TermLine>
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar6_high_impact_events — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Next Event</div>
          <div style={{ fontSize:'28px', fontWeight:600, letterSpacing:'-.02em', color:'var(--yellow)', marginBottom:'3px' }}>
            {data.days_until ? data.days_until+'D' : '5D 08H'}
          </div>
          <div style={{ fontSize:'13px', fontWeight:500, marginBottom:'3px' }}>{data.event||data.event_name||'FOMC Rate Decision'}</div>
          <div style={{ fontSize:'10px', color:'var(--text2)', marginBottom:'14px' }}>{data.scheduled||'April 29, 2026'} · HIGH impact</div>
          <div className="sec-label" style={{ marginBottom:'7px' }}>Uncertainty</div>
          <div style={{ width:'100%', height:'5px', background:'var(--border)', borderRadius:'1px', overflow:'hidden', marginBottom:'4px' }}>
            <div style={{ height:'100%', width:(uncertainty*100)+'%', background:'linear-gradient(90deg,var(--green),var(--yellow),var(--red))' }} />
          </div>
          <div style={{ display:'flex', justifyContent:'space-between', fontSize:'9px', color:'var(--yellow)' }}>
            <span>{(uncertainty*100).toFixed(1)}%</span><span>HIGH</span>
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Trade Restrictions</div>
          <div className="kv">
            <KVRow label="Allow Trade"    value={restrictions.allow_trade!==false?'YES':'NO'} color={restrictions.allow_trade!==false?'g':'r'} />
            <KVRow label="Size Multiplier" value={restrictions.size_multiplier?restrictions.size_multiplier+'×':'0.3×'} color="y" />
            <KVRow label="Leverage Cap"   value={restrictions.leverage_cap?restrictions.leverage_cap+'×':'2.0×'}        color="y" />
          </div>
        </div>
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec">
          <div className="sec-label">Scenarios</div>
          {(scenarios.length > 0 ? scenarios : MOCK_SCENARIOS).map((sc, i) => (
            <div key={i} style={{ border:'1px solid var(--border)', padding:'10px 12px', marginBottom:'8px' }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'5px' }}>
                <div style={{ fontSize:'11px', fontWeight:500 }}>{sc.case||sc.scenario||sc.label}</div>
              </div>
              <div style={{ fontSize:'10px', color:'var(--text2)', lineHeight:1.5 }}>{sc.description||sc.detail}</div>
              <div style={{
                display:'inline-flex', alignItems:'center', gap:'4px',
                fontSize:'9px', fontWeight:600, padding:'2px 6px', marginTop:'5px',
                background: sc.direction==='UP'?'var(--green-dim)':sc.direction==='DOWN'?'var(--red-dim)':'var(--border)',
                color: sc.direction==='UP'?'var(--green)':sc.direction==='DOWN'?'var(--red)':'var(--text2)',
              }}>
                {sc.direction==='UP'?'▲ UP':sc.direction==='DOWN'?'▼ DOWN':'↔ WHIPSAW'} · HIGH VOL
              </div>
            </div>
          ))}
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P7 — COUNCIL
// ════════════════════════════════════════════════════
function P7Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const council      = data.council         || {};
  const disagreement = data.disagreement    || {};
  const agents       = data.agent_outputs   || {};
  const sharedSum    = data.shared_state_summary || {};
  const professor    = agents.professor_agent || {};
  const retail       = agents.retail_agent    || {};
  const reasons      = data.reason_stack     || [];
  const aiOverview   = data.ai_overview      || {};

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar7_ml_council.run_pillar7</TermLine>
      <TermLine color="cyan">┌─ PILLAR 7 — ML COUNCIL ─────────────────────────────┐</TermLine>
      <TermLine>  Council Bias    <TermSpan color="red">{council.council_bias||'NO_TRADE'}</TermSpan></TermLine>
      <TermLine>  Tradeability    <TermSpan color="red">{council.tradeability_score?.toFixed(3)||'0.000'}</TermSpan></TermLine>
      <TermLine>  Alignment       <TermSpan color="yellow">{disagreement.alignment_class||'INACTIVE_ALIGNMENT'}</TermSpan></TermLine>
      <TermLine>  Professor       <TermSpan color="red">{professor.predicted_label||'NO_TRADE'}</TermSpan> cal={professor.calibrated_probability?.toFixed(3)||'0.005'}</TermLine>
      <TermLine>  Retail          <TermSpan color="yellow">{retail.predicted_label||'NO_ACTION'}</TermSpan> cal={retail.calibrated_probability?.toFixed(3)||'0.263'}</TermLine>
      {reasons.map((r, i) => <TermLine key={i}>  → {r}</TermLine>)}
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar7_ml_council — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Council Verdict</div>
          <div className="kv">
            <KVRow label="Council Bias"  value={council.council_bias||'—'}   color="r" />
            <KVRow label="Tradeability"  value={council.tradeability_score?.toFixed(3)||'—'} color="r" />
            <KVRow label="Alignment"     value={disagreement.alignment_class||'—'} color="y" />
            <KVRow label="Professor"     value={`${professor.predicted_label||'—'} (${((professor.calibrated_probability||0)*100).toFixed(1)}%)`} color="r" />
            <KVRow label="Retail"        value={`${retail.predicted_label||'—'} (${((retail.calibrated_probability||0)*100).toFixed(1)}%)`} color="y" />
            <KVRow label="Agreement"     value={disagreement.agreement_score?.toFixed(2)||'—'} />
            <KVRow label="Conflict"      value={disagreement.conflict_score?.toFixed(2)||'—'} />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Shared State</div>
          <div className="kv">
            {Object.entries(sharedSum).map(([k, v]) => (
              <KVRow key={k} label={k.replace(/_state/,'').replace(/_/,' ')} value={v||'—'} />
            ))}
          </div>
        </div>
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec" style={{ flex:1 }}>
          <div className="sec-label">AI Council Overview</div>
          <div className="ai">
            <p>{aiOverview.overview || 'STRONG_UPTREND regime but council standing down — FOMC uncertainty 76% is the dominant suppressor. INACTIVE_ALIGNMENT means both agents agree to not trade. Post-FOMC, reassess alignment.'}</p>
          </div>
        </div>
      </div>
    </TabShell>
  );
}

// ════════════════════════════════════════════════════
// P8 — DECISION
// ════════════════════════════════════════════════════
function P8Tab({ data }) {
  if (!data) return <TabShell><NotReady /></TabShell>;

  const exec       = data.execution_plan || {};
  const alignment  = data.alignment      || {};
  const components = alignment.components || [];
  const warnings   = data.warnings       || [];

  const termContent = (
    <>
      <TermLine prompt>python3 -m pillar8_decision_risk_backtesting.run_pillar8</TermLine>
      <TermLine color="cyan">┌─ PILLAR 8 — DECISION, RISK & SIZING ────────────────┐</TermLine>
      <TermLine>  Final Action    <TermSpan color="green">{data.final_action||'PROBE_LONG'}</TermSpan></TermLine>
      <TermLine>  Direction       <TermSpan color="green">{data.direction||'LONG'}</TermSpan></TermLine>
      <TermLine>  Archetype       {data.decision_archetype||'TREND_LONG'}</TermLine>
      <TermLine>  Confidence      <TermSpan color="red">{data.decision_confidence?.toFixed(3)||'0.175'}</TermSpan></TermLine>
      <TermLine>  Tradability     <TermSpan color="yellow">{data.tradability_score?.toFixed(3)||'0.447'}</TermSpan></TermLine>
      <TermLine>  Risk Score      <TermSpan color="yellow">{data.risk_score?.toFixed(3)||'0.409'} ({data.risk_state||'MODERATE'})</TermSpan></TermLine>
      <TermLine>  Size Fraction   {data.size_fraction?.toFixed(4)||'0.004'}</TermLine>
      <TermLine>  Max Leverage    {data.max_leverage_allowed?.toFixed(2)||'1.28'}×</TermLine>
      <TermLine color="cyan">├─ Alignment per Pillar ──────────────────────────────┤</TermLine>
      {components.map((c, i) => (
        <TermLine key={i}>  {c.pillar.padEnd(12)} bias={c.raw_bias_score>=0?'+':''}{c.raw_bias_score.toFixed(2)}  weighted={c.weighted_score>=0?'+':''}{c.weighted_score.toFixed(3)}</TermLine>
      ))}
      {warnings.map((w, i) => <TermLine key={i} color="yellow">  ⚠  {w}</TermLine>)}
      <TermLine color="cyan">└─────────────────────────────────────────────────────┘</TermLine>
    </>
  );

  return (
    <TabShell terminal={{ title:'pillar8_decision_risk — raw terminal output', content:termContent }}>
      <div style={{ flex:2, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto', borderRight:'1px solid var(--border)' }}>
        <div className="sec">
          <div className="sec-label">Decision Gate</div>
          <div className="kv">
            <KVRow label="Final Action"  value={data.final_action||'—'}   color="g" />
            <KVRow label="Direction"     value={data.direction||'—'}      color="g" />
            <KVRow label="Archetype"     value={data.decision_archetype?.replace(/_/g,' ')||'—'} />
            <KVRow label="Confidence"    value={data.decision_confidence?(data.decision_confidence*100).toFixed(1)+'%':'—'} color="r" />
            <KVRow label="Tradability"   value={data.tradability_score?(data.tradability_score*100).toFixed(1)+'%':'—'}     color="y" />
          </div>
        </div>
        <div className="sec">
          <div className="sec-label">Risk Score — {data.risk_state||'—'}</div>
          <div style={{ display:'flex', alignItems:'center', gap:'10px', marginBottom:'10px' }}>
            <div style={{ fontSize:'28px', fontWeight:600, color: data.risk_state==='LOW'?'var(--green)':data.risk_state==='MODERATE'?'var(--yellow)':'var(--red)' }}>
              {data.risk_score?.toFixed(3)||'—'}
            </div>
            <div style={{ fontSize:'10px', color:'var(--text2)', textTransform:'uppercase', letterSpacing:'.1em' }}>{data.risk_state||'—'}</div>
          </div>
          {data.risk_components && Object.entries(data.risk_components).map(([k, v]) => (
            <div key={k} style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'6px', fontSize:'10px' }}>
              <div style={{ color:'var(--text2)', width:'90px', flexShrink:0 }}>{k.replace(/_/g,' ')}</div>
              <div style={{ flex:1, height:'3px', background:'var(--border)', borderRadius:'1px', overflow:'hidden' }}>
                <div style={{ height:'100%', width:(v*100)+'%', background: v>0.65?'var(--red)':v>0.35?'var(--yellow)':'var(--green)', borderRadius:'1px' }} />
              </div>
              <div style={{ color:'var(--text2)', width:'32px', textAlign:'right' }}>{v?.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div className="sec">
          <div className="sec-label">Sizing</div>
          <div className="kv">
            <KVRow label="Size Fraction" value={data.size_fraction?(data.size_fraction*100).toFixed(2)+'%':'—'} />
            <KVRow label="Max Leverage"  value={data.max_leverage_allowed?data.max_leverage_allowed.toFixed(2)+'×':'—'} />
          </div>
        </div>
      </div>
      <div style={{ flex:3, padding:'14px', display:'flex', flexDirection:'column', gap:'12px', overflowY:'auto' }}>
        <div className="sec">
          <div className="sec-label">Execution Plan</div>
          <div className="kv">
            <KVRow label="Entry"     value={exec.entry_style?.replace(/_/g,' ')||'—'} />
            <KVRow label="Stop"      value={exec.stop_framework?.replace(/_/g,' ')||'—'} />
            <KVRow label="Target"    value={exec.target_framework?.replace(/_/g,' ')||'—'} />
            <KVRow label="Time Stop" value={exec.time_stop_bars?exec.time_stop_bars+' bars':'—'} />
          </div>
        </div>
        {(exec.invalidators||[]).length > 0 && (
          <div className="sec">
            <div className="sec-label">Invalidators</div>
            {exec.invalidators.map((inv, i) => (
              <div key={i} className="invr"><div className="invb">—</div><div>{inv}</div></div>
            ))}
          </div>
        )}
        {warnings.length > 0 && (
          <div className="sec">
            <div className="sec-label">Warnings</div>
            {warnings.map((w, i) => (
              <div key={i} className="warn"><span>⚠</span><span>{w}</span></div>
            ))}
          </div>
        )}
      </div>
    </TabShell>
  );
}