import { useEffect, useRef } from 'react';

export default function Overview({ data }) {
  const tvRef = useRef(null);
  const tvLoaded = useRef(false);

  const p6 = data.pillar6;
  const p7 = data.pillar7;
  const p8 = data.pillar8;

  // ── TradingView chart ──────────────────────────────────────────────────────
  useEffect(() => {
    if (tvLoaded.current) return;
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      if (window.TradingView && tvRef.current) {
        new window.TradingView.widget({
          container_id: 'tv_overview',
          symbol: 'BINANCE:BTCUSDT',
          interval: '60',
          timezone: 'UTC',
          theme: 'dark',
          style: '1',
          locale: 'en',
          toolbar_bg: '#111',
          enable_publishing: false,
          hide_side_toolbar: false,
          allow_symbol_change: false,
          save_image: false,
          backgroundColor: '#0d0d0d',
          gridColor: '#181818',
          width: '100%',
          height: '100%',
          withdateranges: true,
          hide_legend: false,
        });
        tvLoaded.current = true;
      }
    };
    document.head.appendChild(script);
  }, []);

  // ── Derived values from P8 ─────────────────────────────────────────────────
  const finalAction     = p8?.final_action     || '—';
  const direction       = p8?.direction        || '—';
  const archetype       = p8?.decision_archetype || '—';
  const riskScore       = p8?.risk_score;
  const riskState       = p8?.risk_state       || '—';
  const sizeFraction    = p8?.size_fraction;
  const maxLeverage     = p8?.max_leverage_allowed;
  const confidence      = p8?.decision_confidence;
  const tradability     = p8?.tradability_score;
  const vetoes          = p8?.vetoes           || [];
  const warnings        = p8?.warnings         || [];
  const execPlan        = p8?.execution_plan   || {};
  const invalidators    = execPlan.invalidators || [];

  // ── Derived from P7 ────────────────────────────────────────────────────────
  const alignment       = p8?.alignment        || {};
  const components      = alignment.components || [];

  // ── Pillar signals from cached data ───────────────────────────────────────
  const signals = [
    { key:'p1', label:'P1 Sent',  state: getSentimentState(data.pillar1),  color: 'dim',    width:'50%', arrow:'→' },
    { key:'p2', label:'P2 Mem',   state: getMemoryState(data.pillar2),     color: 'r',      width:'60%', arrow:'↓' },
    { key:'p3', label:'P3 Struct',state: getStructureState(data.pillar3),  color: 'dim',    width:'42%', arrow:'→' },
    { key:'p4', label:'P4 Cndl',  state: getCandleState(data.pillar4),     color: 'g',      width:'40%', arrow:'↑' },
    { key:'p5', label:'P5 Regime',state: getRegimeState(data.pillar5),     color: 'g',      width:'90%', arrow:'↑↑'},
    { key:'p6', label:'P6 Event', state: getEventState(data.pillar6),      color: 'y',      width:'76%', arrow:'⚡'},
    { key:'p7', label:'P7 Cncl',  state: getCouncilState(data.pillar7),    color: 'dim',    width:'22%', arrow:'—' },
    { key:'p8', label:'P8 Dcsn',  state: finalAction !== '—' ? finalAction.replace('_',' ') : '—', color:'g', width:'45%', arrow:'↑' },
  ];

  const actionColor = getActionColor(finalAction);
  const riskColor   = getRiskColor(riskState);

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%', overflow:'hidden' }}>

      {/* TOP: Chart | Verdict | Signals */}
      <div style={{ display:'flex', flex:1, minHeight:0, overflow:'hidden' }}>

        {/* CHART */}
        <div style={{ flex:5, borderRight:'1px solid var(--border)', overflow:'hidden' }}>
          <div id="tv_overview" ref={tvRef} style={{ width:'100%', height:'100%' }} />
        </div>

        {/* VERDICT */}
        <div style={{ flex:3, borderRight:'1px solid var(--border)', padding:'14px', overflowY:'auto', display:'flex', flexDirection:'column', gap:'12px' }}>

          <div className="sec">
            <div className="sec-label">Verdict</div>
            {p8 ? (
              <>
                <div style={{ fontSize:'24px', fontWeight:600, letterSpacing:'-0.02em', color:actionColor, marginBottom:'3px' }}>
                  {finalAction.replace(/_/g,' ')}
                </div>
                <div style={{ fontSize:'10px', color:'var(--text2)', marginBottom:'14px' }}>
                  {archetype.replace(/_/g,' ')} · {riskState} Risk
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'10px' }}>
                  <VItem label="Risk"       value={riskState}                              color={riskColor}   tip={riskScore?.toFixed(3)} />
                  <VItem label="Size"       value={sizeFraction ? (sizeFraction*100).toFixed(1)+'%' : '—'}    tip={sizeFraction?.toFixed(4)} />
                  <VItem label="Leverage"   value={maxLeverage ? maxLeverage.toFixed(2)+'×' : '—'}            tip="max allowed" />
                  <VItem label="Confidence" value={getConfLabel(confidence)}               color={getConfColor(confidence)} tip={confidence?.toFixed(3)} />
                </div>
              </>
            ) : <NotReady label="P8 computing..." />}
          </div>

          <div className="sec">
            <div className="sec-label">Execution</div>
            <div className="kvr"><div className="kvk">Entry</div><div className="kvv">{execPlan.entry_style?.replace(/_/g,' ') || '—'}</div></div>
            <div className="kvr"><div className="kvk">Stop</div><div className="kvv">{execPlan.stop_framework?.replace(/_/g,' ') || '—'}</div></div>
            <div className="kvr"><div className="kvk">Target</div><div className="kvv">{execPlan.target_framework?.replace(/_/g,' ') || '—'}</div></div>
            <div className="kvr"><div className="kvk">Time stop</div><div className="kvv">{execPlan.time_stop_bars ? execPlan.time_stop_bars + ' bars' : '—'}</div></div>
          </div>

          {warnings.length > 0 && (
            <div className="sec">
              <div className="sec-label">Warnings</div>
              {warnings.map((w, i) => (
                <div key={i} className="warn"><span>⚠</span><span>{w}</span></div>
              ))}
            </div>
          )}

        </div>

        {/* SIGNALS + ALIGNMENT */}
        <div style={{ flex:2, padding:'14px', overflowY:'auto', display:'flex', flexDirection:'column', gap:'12px' }}>

          <div className="sec">
            <div className="sec-label">Pillar Signals</div>
            {signals.map(sig => (
              <SignalRow key={sig.key} {...sig} />
            ))}
          </div>

          <div className="sec">
            <div className="sec-label">Alignment</div>
            {components.length > 0 ? (
              components.map((c, i) => (
                <AlignRow
                  key={i}
                  label={c.pillar}
                  score={c.weighted_score}
                  direction={c.weighted_score > 0.05 ? 'long' : c.weighted_score < -0.05 ? 'short' : null}
                />
              ))
            ) : (
              ['Sentiment','Memory','Regime','Events','Council'].map(l => (
                <AlignRow key={l} label={l} score={0} direction={null} />
              ))
            )}
          </div>

        </div>
      </div>

      {/* BOTTOM STRIP: Invalidators + Warnings */}
      <div style={{ display:'flex', borderTop:'1px solid var(--border)', flexShrink:0 }}>
        <div style={{ flex:1, padding:'10px 14px', borderRight:'1px solid var(--border)' }}>
          <div style={{ fontSize:'8px', letterSpacing:'.15em', color:'var(--text3)', textTransform:'uppercase', marginBottom:'7px' }}>Invalidators</div>
          {invalidators.length > 0
            ? invalidators.map((inv, i) => (
                <div key={i} className="invr"><div className="invb">—</div><div>{inv}</div></div>
              ))
            : ['Edge flips away from LONG across pillars','Breakout acceptance fails after entry','Event uncertainty expands further'].map((inv,i) => (
                <div key={i} className="invr"><div className="invb">—</div><div>{inv}</div></div>
              ))
          }
        </div>
        <div style={{ flex:2, padding:'10px 14px' }}>
          <div style={{ fontSize:'8px', letterSpacing:'.15em', color:'var(--text3)', textTransform:'uppercase', marginBottom:'7px' }}>Active Warnings</div>
          <div style={{ display:'flex', gap:'8px', flexWrap:'wrap' }}>
            {warnings.length > 0
              ? warnings.map((w, i) => (
                  <div key={i} className="warn" style={{ margin:0, flex:'1', minWidth:'180px' }}><span>⚠</span><span>{w}</span></div>
                ))
              : <div style={{ color:'var(--text3)', fontSize:'10px' }}>No active warnings</div>
            }
          </div>
        </div>
      </div>

    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function VItem({ label, value, color, tip }) {
  return (
    <div>
      <div style={{ fontSize:'8px', color:'var(--text3)', letterSpacing:'.1em', textTransform:'uppercase', marginBottom:'2px' }}>{label}</div>
      <div title={tip} style={{ fontSize:'13px', fontWeight:500, color: color || 'var(--text)', cursor: tip ? 'help' : 'default' }}>
        {value || '—'}
      </div>
    </div>
  );
}

function SignalRow({ label, state, color, width, arrow }) {
  const barColor = color === 'g' ? 'var(--green)' : color === 'r' ? 'var(--red)' : color === 'y' ? 'var(--yellow)' : 'var(--text3)';
  const textColor = color === 'g' ? 'var(--green)' : color === 'r' ? 'var(--red)' : color === 'y' ? 'var(--yellow)' : 'var(--text2)';
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'6px 0', borderBottom:'1px solid var(--border)', cursor:'pointer' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
        <div style={{ fontSize:'9px', color:'var(--text3)', letterSpacing:'.1em', textTransform:'uppercase', width:'62px' }}>{label}</div>
        <div style={{ width:'50px', height:'2px', background:'var(--border)', borderRadius:'1px', overflow:'hidden' }}>
          <div style={{ height:'100%', width, background:barColor, borderRadius:'1px' }} />
        </div>
      </div>
      <div style={{ fontSize:'10px', fontWeight:500, color:textColor }}>{state}</div>
      <div style={{ fontSize:'12px', width:'14px', textAlign:'center', color:'var(--text3)' }}>{arrow}</div>
    </div>
  );
}

function AlignRow({ label, score, direction }) {
  const fillWidth = Math.abs(score) * 45 + '%';
  const fillColor = direction === 'long' ? 'var(--green)' : direction === 'short' ? 'var(--red)' : 'transparent';
  const lblColor  = direction === 'long' ? 'var(--green)' : direction === 'short' ? 'var(--red)' : 'var(--text3)';
  return (
    <div style={{ display:'flex', alignItems:'center', padding:'5px 0', borderBottom:'1px solid var(--border)', fontSize:'10px' }}>
      <div style={{ color:'var(--text3)', width:'64px', textTransform:'uppercase', fontSize:'9px', letterSpacing:'.08em' }}>{label}</div>
      <div style={{ flex:1, height:'2px', background:'var(--border)', borderRadius:'1px', position:'relative', margin:'0 8px' }}>
        <div style={{ position:'absolute', left:'50%', top:'-2px', width:'1px', height:'6px', background:'var(--border2)', transform:'translateX(-50%)' }} />
        {direction === 'long'  && <div style={{ position:'absolute', left:'50%',  top:0, height:'100%', width:fillWidth, background:fillColor, borderRadius:'1px' }} />}
        {direction === 'short' && <div style={{ position:'absolute', right:'50%', top:0, height:'100%', width:fillWidth, background:fillColor, borderRadius:'1px' }} />}
      </div>
      <div style={{ width:'70px', textAlign:'right', color:lblColor }}>
        {direction === 'long' ? '↑ LONG' : direction === 'short' ? '↓ SHORT' : '·'}
      </div>
    </div>
  );
}

function NotReady({ label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', color:'var(--text3)', fontSize:'10px', padding:'8px 0' }}>
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}

// ── Data extractors ──────────────────────────────────────────────────────────
function getSentimentState(p1) {
  if (!p1) return '—';
  return p1.aggregate_sentiment?.label || 'NEUTRAL';
}
function getMemoryState(p2) {
  if (!p2) return '—';
  return p2.memory_summary?.memory_bias?.replace('_BIAS','') || 'MEAN REV';
}
function getStructureState(p3) {
  if (!p3) return '—';
  const s = p3.structure_state;
  if (typeof s === 'object') return s.market_structure || 'MIXED';
  return s || 'MIXED';
}
function getCandleState(p4) {
  if (!p4) return '—';
  return p4.candle_summary?.dominant_intent?.replace('_CANDIDATE','') || 'BUY ABS';
}
function getRegimeState(p5) {
  if (!p5) return '—';
  return p5.regime_summary?.directional_regime?.replace('_',' ') || 'STRONG UP';
}
function getEventState(p6) {
  if (!p6) return '—';
  const event = p6.event || p6.event_name || 'FOMC';
  return event.slice(0,8).toUpperCase();
}
function getCouncilState(p7) {
  if (!p7) return '—';
  return p7.council?.council_bias?.replace('_','-') || 'NO TRD';
}
function getActionColor(action) {
  if (!action || action === '—') return 'var(--text)';
  if (action.includes('LONG'))  return 'var(--green)';
  if (action.includes('SHORT')) return 'var(--red)';
  if (action === 'WATCHLIST')   return 'var(--yellow)';
  return 'var(--red)';
}
function getRiskColor(state) {
  if (state === 'LOW')      return 'var(--green)';
  if (state === 'MODERATE') return 'var(--yellow)';
  if (state === 'HIGH')     return 'var(--red)';
  if (state === 'EXTREME')  return 'var(--red)';
  return 'var(--text)';
}
function getConfLabel(v) {
  if (v == null) return '—';
  if (v >= 0.65) return 'HIGH';
  if (v >= 0.45) return 'MODERATE';
  return 'LOW';
}
function getConfColor(v) {
  if (v == null) return 'var(--text)';
  if (v >= 0.65) return 'var(--green)';
  if (v >= 0.45) return 'var(--yellow)';
  return 'var(--red)';
}