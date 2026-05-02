import { useEffect, useRef } from 'react';

export default function Overview({ data }) {
  const tvRef    = useRef(null);
  const tvLoaded = useRef(false);

  const p6 = data.pillar6;
  const p7 = data.pillar7;
  const p8 = data.pillar8;

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

  const finalAction  = p8?.final_action        || '—';
  const archetype    = p8?.decision_archetype   || '—';
  const riskScore    = p8?.risk_score;
  const riskState    = p8?.risk_state           || '—';
  const sizeFraction = p8?.size_fraction;
  const maxLeverage  = p8?.max_leverage_allowed;
  const confidence   = p8?.decision_confidence;
  const warnings     = p8?.warnings             || [];
  const execPlan     = p8?.execution_plan        || {};
  const invalidators = execPlan.invalidators     || [];
  const alignment    = p8?.alignment             || {};
  const components   = alignment.components      || [];

  const signals = [
    { key:'p1', label:'P1 Sent',   state: getSentimentState(data.pillar1),  color: getSentimentColor(data.pillar1),  arrow:'→' },
    { key:'p2', label:'P2 Mem',    state: getMemoryState(data.pillar2),      color: 'r',   arrow:'↓' },
    { key:'p3', label:'P3 Struct', state: getStructureState(data.pillar3),   color: 'dim', arrow:'→' },
    { key:'p4', label:'P4 Cndl',   state: getCandleState(data.pillar4),      color: 'g',   arrow:'↑' },
    { key:'p5', label:'P5 Regime', state: getRegimeState(data.pillar5),      color: 'g',   arrow:'↑↑'},
    { key:'p6', label:'P6 Event',  state: getEventState(data.pillar6),       color: 'y',   arrow:'⚡'},
    { key:'p7', label:'P7 Cncl',   state: getCouncilState(data.pillar7),     color: 'dim', arrow:'—' },
    { key:'p8', label:'P8 Dcsn',   state: finalAction !== '—' ? finalAction.replace(/_/g,' ') : '—', color:'g', arrow:'↑' },
  ];

  const actionColor = getActionColor(finalAction);
  const riskColor   = getRiskColor(riskState);

  return (
    // KEY FIX: width:100% + overflow:hidden on root ensures full width
    <div style={{ display:'flex', flexDirection:'column', height:'100%', width:'100%', overflow:'hidden' }}>

      {/* TOP ROW: Chart | Verdict | Signals — flex row, full width */}
      <div style={{ display:'flex', flex:1, minHeight:0, overflow:'hidden', width:'100%' }}>

        {/* CHART — takes majority of space */}
        <div style={{ flex:5, borderRight:'1px solid var(--border)', overflow:'hidden', minWidth:0 }}>
          <div id="tv_overview" ref={tvRef} style={{ width:'100%', height:'100%' }} />
        </div>

        {/* VERDICT */}
        <div style={{ flex:3, borderRight:'1px solid var(--border)', padding:'14px', overflowY:'auto', display:'flex', flexDirection:'column', gap:'12px', minWidth:0 }}>
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
                  <VItem label="Risk"       value={riskState}                                                         color={riskColor}          tip={riskScore?.toFixed(3)} />
                  <VItem label="Size"       value={sizeFraction ? (sizeFraction*100).toFixed(1)+'%' : '—'}            tip={sizeFraction?.toFixed(4)} />
                  <VItem label="Leverage"   value={maxLeverage  ? maxLeverage.toFixed(2)+'×'       : '—'}            tip="max allowed" />
                  <VItem label="Confidence" value={getConfLabel(confidence)}                                          color={getConfColor(confidence)} tip={confidence?.toFixed(3)} />
                </div>
              </>
            ) : <NotReady label="P8 computing..." />}
          </div>

          <div className="sec">
            <div className="sec-label">Execution</div>
            <div className="kvr"><div className="kvk">Entry</div><div className="kvv">{execPlan.entry_style?.replace(/_/g,' ')    || '—'}</div></div>
            <div className="kvr"><div className="kvk">Stop</div><div className="kvv">{execPlan.stop_framework?.replace(/_/g,' ')  || '—'}</div></div>
            <div className="kvr"><div className="kvk">Target</div><div className="kvv">{execPlan.target_framework?.replace(/_/g,' ') || '—'}</div></div>
            <div className="kvr"><div className="kvk">Time stop</div><div className="kvv">{execPlan.time_stop_bars ? execPlan.time_stop_bars+' bars' : '—'}</div></div>
          </div>

          {warnings.length > 0 && (
            <div className="sec">
              <div className="sec-label">Warnings</div>
              {warnings.map((w, i) => <div key={i} className="warn"><span>⚠</span><span>{w}</span></div>)}
            </div>
          )}
        </div>

        {/* SIGNALS + ALIGNMENT */}
        <div style={{ flex:2, padding:'14px', overflowY:'auto', display:'flex', flexDirection:'column', gap:'12px', minWidth:0 }}>
          <div className="sec">
            <div className="sec-label">Pillar Signals</div>
            {signals.map(sig => <SignalRow key={sig.key} {...sig} />)}
          </div>
          <div className="sec">
            <div className="sec-label">Alignment</div>
            {components.length > 0
              ? components.map((c, i) => (
                  <AlignRow key={i} label={c.pillar} score={c.weighted_score}
                    direction={c.weighted_score > 0.05 ? 'long' : c.weighted_score < -0.05 ? 'short' : null} />
                ))
              : ['Sentiment','Memory','Structure','Regime','Events','Council'].map(l => (
                  <AlignRow key={l} label={l} score={0} direction={null} />
                ))
            }
          </div>
        </div>
      </div>

      {/* BOTTOM STRIP */}
      <div style={{ display:'flex', borderTop:'1px solid var(--border)', flexShrink:0 }}>
        <div style={{ flex:1, padding:'10px 14px', borderRight:'1px solid var(--border)' }}>
          <div style={{ fontSize:'8px', letterSpacing:'.15em', color:'var(--text3)', textTransform:'uppercase', marginBottom:'7px' }}>Invalidators</div>
          {(invalidators.length > 0 ? invalidators : [
            'Edge flips away from LONG across pillars',
            'Breakout acceptance fails after entry',
            'Event uncertainty expands further',
          ]).map((inv, i) => (
            <div key={i} className="invr"><div className="invb">—</div><div>{inv}</div></div>
          ))}
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

// ── Sub-components ────────────────────────────────────────────────────────────

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

function SignalRow({ label, state, color, arrow }) {
  const barColor  = color === 'g' ? 'var(--green)' : color === 'r' ? 'var(--red)' : color === 'y' ? 'var(--yellow)' : 'var(--text3)';
  const textColor = color === 'g' ? 'var(--green)' : color === 'r' ? 'var(--red)' : color === 'y' ? 'var(--yellow)' : 'var(--text2)';
  const barWidth  = color === 'g' ? '75%' : color === 'r' ? '55%' : color === 'y' ? '65%' : '30%';
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'6px 0', borderBottom:'1px solid var(--border)' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
        <div style={{ fontSize:'9px', color:'var(--text3)', letterSpacing:'.1em', textTransform:'uppercase', width:'62px' }}>{label}</div>
        <div style={{ width:'50px', height:'2px', background:'var(--border)', borderRadius:'1px', overflow:'hidden' }}>
          <div style={{ height:'100%', width:barWidth, background:barColor, borderRadius:'1px' }} />
        </div>
      </div>
      <div style={{ fontSize:'10px', fontWeight:500, color:textColor, flex:1, textAlign:'right', paddingRight:'4px', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{state}</div>
      <div style={{ fontSize:'12px', width:'14px', textAlign:'center', color:'var(--text3)', flexShrink:0 }}>{arrow}</div>
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

// ── Terminal output renderer ──────────────────────────────────────────────────
// Used in each pillar tab to show the rich terminal-style output

export function TerminalOutput({ lines }) {
  return (
    <div style={{
      fontFamily: 'var(--mono)',
      fontSize: '10px',
      lineHeight: '1.7',
      color: '#ccc',
      background: '#000',
      padding: '12px 14px',
      overflowY: 'auto',
      flex: 1,
      minHeight: 0,
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
    }}>
      {lines.map((line, i) => {
        // Color coding based on content
        let color = '#ccc';
        if (line.startsWith('╭') || line.startsWith('╰') || line.startsWith('│')) color = '#4af';
        else if (line.includes('⚠')) color = '#f0c040';
        else if (line.match(/BULLISH|UPTREND|STRONG_UP|BUY|LONG|POSITIVE|FAVORED|LOW risk/i)) color = '#3ddc84';
        else if (line.match(/BEARISH|DOWNTREND|SELL|SHORT|NEGATIVE|NOT_FAVORED|HIGH risk/i)) color = '#ff4d4d';
        else if (line.match(/NEUTRAL|MODERATE|MIXED|BALANCED/i)) color = '#f0c040';
        else if (line.startsWith('  ▲')) color = '#3ddc84';
        else if (line.startsWith('  ▼')) color = '#ff4d4d';
        else if (line.match(/━+/)) color = '#333';
        else if (line.match(/AI .* Overview|Institutional Macro|Risk Flags|Regime Explanation|Context Memory/)) color = '#888';
        return <div key={i} style={{ color }}>{line}</div>;
      })}
    </div>
  );
}

// ── Build terminal lines from pillar data ─────────────────────────────────────

export function buildP1Lines(d) {
  if (!d) return ['  Loading P1 — Sentiment...'];
  const agg = d.aggregate_sentiment || {};
  const src = d.source_distribution || {};
  const articles = d.raw_articles || [];
  const drivers = d.drivers || [];
  const summary = d.institutional_summary || [];
  const lines = [
    '╭────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 1 DATA │',
    '╰────────────────────────────────────────╯',
    '',
    '  Metric                                    Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Sentiment State        ${(agg.label || 'NEUTRAL').toUpperCase()}`,
    `  Model Confidence       ${((agg.confidence || 0) * 100).toFixed(1)}%`,
    `  Articles Analyzed      ${d.article_count || articles.length || 0}`,
    `  Last Updated           ${d.timestamp || '—'}`,
    '',
    '  Institutional Sources',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
  ];
  Object.entries(src).forEach(([s, n]) => lines.push(`  ${s.padEnd(24)} ${n}`));
  if (drivers.length) {
    lines.push('', '  Top Market Drivers:');
    drivers.slice(0,5).forEach(d => lines.push(`   » ${d}`));
  }
  if (summary.length) {
    lines.push('', '  Institutional Macro Summary:');
    (Array.isArray(summary) ? summary : [summary]).forEach(s => lines.push(`   ● ${s}`));
  }
  if (articles.length) {
    lines.push('', '  Raw Extracted Headlines:');
    articles.slice(0,8).forEach(a => lines.push(`   • ${a.headline || a.title || ''}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP2Lines(d) {
  if (!d) return ['  Loading P2 — Market Memory...'];
  const ms   = d.memory_summary    || {};
  const sig  = d.current_state_signature || {};
  const fwd  = d.forward_outcomes  || {};
  const dist = d.distribution_diagnostics || {};
  const stab = d.stability_diagnostics || {};
  const ctx  = d.context_memory    || {};
  const flags = d.risk_flags       || [];
  const ai   = d.ai_overview;
  const aiText = typeof ai === 'string' ? ai : ai?.overview || ai?.headline || '';

  const lines = [
    '╭──────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 2 — MARKET MEMORY ENGINE │',
    '╰──────────────────────────────────────────────────────────╯',
    '',
    '  Metric                              Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Memory Bias                         ${ms.memory_bias || '—'}`,
    `  Match Quality                       ${ms.match_quality || '—'}`,
    `  Sample Size                         ${ms.sample_size || '—'}`,
    `  Headline Confidence                 ${ms.headline_confidence || '—'}`,
    `  Timestamp (UTC)                     ${d.timestamp || '—'}`,
    '',
    '  Current State Signature',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
  ];
  Object.entries(sig).forEach(([k, v]) => lines.push(`  ${k.replace(/_/g,' ').padEnd(20)}  ${v}`));
  lines.push(
    '',
    '  Forward Outcome Probabilities',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Next 3-Bar Up           ${fwd.next_3bar_up || fwd['3bar_up_prob'] || '—'}`,
    `  Next 6-Bar Up           ${fwd.next_6bar_up || fwd['6bar_up_prob'] || '—'}`,
    `  Continuation Prob       ${fwd.continuation_prob || '—'}`,
    `  Reversal Prob           ${fwd.reversal_prob || '—'}`,
    `  Mean Reversion Prob     ${fwd.mean_reversion_prob || '—'}`,
    `  Mean MFE 6-Bar          ${fwd.mean_mfe_6bar || '—'}`,
    `  Mean MAE 6-Bar          ${fwd.mean_mae_6bar || '—'}`,
    '',
    '  Temporal Stability',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Temporal Stability: ${stab.temporal_stability_score ?? '—'}   Regime Dependency: ${stab.regime_dependency_score ?? '—'}`,
  );
  if (Object.keys(ctx).length) {
    lines.push('', '  Context Memory Tendencies:');
    Object.entries(ctx).forEach(([k, v]) => lines.push(`  ${k.padEnd(16)} ${v}`));
  }
  if (flags.length) {
    lines.push('', '  Risk Flags:');
    flags.forEach(f => lines.push(`  ⚠  ${f}`));
  }
  if (aiText) {
    lines.push('', '  AI Memory Overview (groq)');
    aiText.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP3Lines(d) {
  if (!d) return ['  Loading P3 — Structure & Liquidity...'];
  const sb  = d.structure_state || {};
  const liq = d.liquidity_levels || {};
  const sum = d.structure_liquidity_summary || {};
  const trap = d.trap_detection || {};
  const lr  = d.liquidation_risk || {};
  const targets = d.liquidity_targets || [];
  const flags = d.risk_flags || [];
  const ai  = d.ai_overview?.overview || '';
  const struct = typeof sb === 'object' ? sb.market_structure : sb;

  const lines = [
    '╭──────────────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 3 — STRUCTURE & LIQUIDITY ENGINE │',
    '╰──────────────────────────────────────────────────────────────────╯',
    '',
    '  Metric                              Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Market Structure                    ${struct || '—'}`,
    `  Range State                         ${sb.range_state || '—'}`,
    `  Compression State                   ${sb.compression_state || '—'}`,
    `  Dominant Liquidity Side             ${sum.dominant_liquidity_side || '—'}`,
    `  Liquidity Environment               ${sum.liquidity_environment || '—'}`,
    `  Trap Risk                           ${sum.trap_risk || '—'}`,
    '',
    '  Liquidity Levels',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Buy-Side Liquidity    $${liq.buy_side_liquidity?.toLocaleString() || '—'}`,
    `  Sell-Side Liquidity   $${liq.sell_side_liquidity?.toLocaleString() || '—'}`,
    `  Nearest Magnet        $${liq.nearest_liquidity_magnet?.toLocaleString() || '—'}`,
    '',
    '  Trap Detection',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Breakout Trap Probability   ${trap.breakout_trap_probability ? (trap.breakout_trap_probability*100).toFixed(1)+'%' : '—'}`,
    `  Breakdown Trap Probability  ${trap.breakdown_trap_probability ? (trap.breakdown_trap_probability*100).toFixed(1)+'%' : '—'}`,
    `  Likely Trap Side            ${trap.likely_trap_side || 'NO_CLEAR_TRAP'}`,
    '',
    '  Liquidation Risk',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Long Liquidation Risk    ${lr.long_liquidation_risk || '—'}`,
    `  Short Liquidation Risk   ${lr.short_liquidation_risk || '—'}`,
    `  Cascade Probability      ${lr.cascade_probability ? (lr.cascade_probability*100).toFixed(1)+'%' : '—'}`,
  ];
  if (targets.length) {
    lines.push('', '  Liquidity Targets (nearest first):');
    targets.slice(0,6).forEach(t => {
      const dir = t.side === 'buy' || t.distance_pct > 0 ? '▲' : '▼';
      lines.push(`  ${dir} $${t.price?.toLocaleString() || '—'}  ${t.distance_pct ? (t.distance_pct > 0 ? '+' : '') + (t.distance_pct*100).toFixed(2)+'%' : ''}`);
    });
  }
  if (flags.length) {
    lines.push('', '  Risk Flags:');
    flags.forEach(f => lines.push(`  ⚠  ${f}`));
  }
  if (ai) {
    lines.push('', '  AI Structure & Liquidity Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP4Lines(d) {
  if (!d) return ['  Loading P4 — Candle Intelligence...'];
  const cs   = d.candle_summary    || {};
  const prs  = d.pressure          || {};
  const abs  = d.absorption        || {};
  const brk  = d.breakout_analysis || {};
  const feat = d.candle_features   || {};
  const flags = d.risk_flags       || [];
  const ai   = d.ai_overview?.overview || '';
  const intents = d.intent_scores  || [];

  const lines = [
    '╭────────────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 4 — CANDLE INTELLIGENCE ENGINE │',
    '╰────────────────────────────────────────────────────────────────╯',
    '',
    '  Metric                              Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Dominant Intent                     ${cs.dominant_intent || '—'}`,
    `  Intent Confidence                   ${cs.intent_confidence ? (cs.intent_confidence*100).toFixed(1)+'%' : '—'}`,
    `  Momentum State                      ${cs.momentum_state || '—'}`,
    `  Control State                       ${cs.control_state || '—'}`,
    `  Expansion State                     ${cs.expansion_state || '—'}`,
    `  Overlap State                       ${cs.overlap_state || '—'}`,
    `  Follow-Through                      ${cs.follow_through_quality || '—'}`,
    `  Exhaustion State                    ${cs.exhaustion_state || 'NONE'}`,
    '',
    '  Pressure Analysis',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Pressure Bias        ${prs.pressure_bias || '—'}`,
    `  Pressure Strength    ${prs.pressure_strength || '—'}`,
    `  Buying Pressure      ${prs.buying_pressure?.toFixed(3) || '—'}`,
    `  Selling Pressure     ${prs.selling_pressure?.toFixed(3) || '—'}`,
    `  Net Pressure         ${prs.net_pressure_score?.toFixed(3) || '—'}`,
    '',
    '  Absorption & Rejection',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Dominant Absorption  ${abs.dominant_absorption || '—'}`,
    `  Dominant Rejection   ${abs.dominant_rejection || '—'}`,
    `  Buy Absorption Score ${abs.buy_absorption_score?.toFixed(3) || '—'}`,
    `  Sell Absorption Score ${abs.sell_absorption_score?.toFixed(3) || '—'}`,
    `  Confidence           ${abs.absorption_confidence?.toFixed(3) || '—'}`,
  ];
  if (intents.length) {
    lines.push('', '  Top Intent Scores:');
    intents.slice(0,5).forEach(it => {
      const bar = '█'.repeat(Math.round((it.score || 0) * 20));
      lines.push(`  ${(it.label || '').padEnd(36)} ${bar.padEnd(12)} ${(it.score || 0).toFixed(3)}`);
    });
  }
  if (flags.length) {
    lines.push('', '  Risk Flags:');
    flags.forEach(f => lines.push(`  ⚠  ${f}`));
  }
  if (ai) {
    lines.push('', '  AI Candle Intelligence Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP5Lines(d) {
  if (!d) return ['  Loading P5 — Regime & Cycle...'];
  const rs    = d.regime_summary        || {};
  const strat = d.strategy_compatibility || {};
  const flags = d.risk_flags            || [];
  const ai    = d.ai_overview?.overview || '';

  const lines = [
    '╭───────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 5 — REGIME & CYCLE ENGINE │',
    '╰───────────────────────────────────────────────────────────╯',
    '',
    '  Metric                           Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Directional Regime               ${rs.directional_regime || '—'}`,
    `  Volatility Regime                ${rs.volatility_regime || '—'}`,
    `  Market State                     ${rs.market_state || '—'}`,
    `  Cycle Phase                      ${rs.cycle_phase || '—'}`,
    `  Confidence Score                 ${d.confidence_score ? (d.confidence_score*100).toFixed(1)+'%' : '—'}`,
    `  Stand Down                       ${d.stand_down ? 'YES — '+d.stand_down_reason : 'NO — CONDITIONS ACCEPTABLE'}`,
    `  Current Session                  ${rs.current_session || d.current_session || '—'}`,
    '',
    '  Price & Key Metrics',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Return 7d                        ${rs.return_7d ? (rs.return_7d*100).toFixed(2)+'%' : '—'}`,
    `  Vol Percentile                   ${rs.volatility_percentile ? (rs.volatility_percentile*100).toFixed(1)+'%' : '—'}`,
    `  MA Order                         ${rs.ma_order || '—'}`,
    `  Swing Structure                  ${rs.swing_structure || '—'}`,
    '',
    '  Strategy Compatibility',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Trend Following    ${strat.trend_following || '—'}`,
    `  Breakout Trading   ${strat.breakout_trading || '—'}`,
    `  Mean Reversion     ${strat.mean_reversion || '—'}`,
  ];
  if (flags.length) {
    lines.push('', '  Risk Flags:');
    flags.forEach(f => lines.push(`  ⚠  ${f}`));
  }
  if (ai) {
    lines.push('', '  AI Regime Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP6Lines(d) {
  if (!d) return ['  Loading P6 — High Impact Events...'];
  const restr = d.trade_restrictions || {};
  const scens = d.scenarios          || [];
  const ai    = d.ai_overview?.overview || '';

  const lines = [
    '╭─────────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 6 — HIGH IMPACT EVENTS      │',
    '╰─────────────────────────────────────────────────────────────╯',
    '',
    '  Metric                           Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Event                            ${d.event || d.event_name || '—'}`,
    `  State                            ${d.state || '—'}`,
    `  Scheduled                        ${d.scheduled || '—'}`,
    `  Days Until                       ${d.days_until ?? '—'}`,
    `  Base Uncertainty                 ${d.base_uncertainty ? (d.base_uncertainty*100).toFixed(1)+'%' : '—'}`,
    '',
    '  Trade Restrictions',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Allow Trade          ${restr.allow_trade !== false ? 'YES' : 'NO'}`,
    `  Size Multiplier      ${restr.size_multiplier ? restr.size_multiplier+'×' : '—'}`,
    `  Leverage Cap         ${restr.leverage_cap ? restr.leverage_cap+'×' : '—'}`,
  ];
  if (scens.length) {
    lines.push('', '  Scenarios:');
    scens.forEach(sc => {
      const dir = sc.direction === 'UP' ? '▲ UP' : sc.direction === 'DOWN' ? '▼ DOWN' : '↔ WHIPSAW';
      lines.push(`  ${dir}  ${sc.case || sc.scenario || sc.label || ''}`);
      if (sc.description || sc.detail) lines.push(`     ${(sc.description || sc.detail || '').substring(0,80)}`);
    });
  }
  if (ai) {
    lines.push('', '  AI Events Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP7Lines(d) {
  if (!d) return ['  Loading P7 — ML Council...'];
  const council = d.council          || {};
  const dis     = d.disagreement     || {};
  const agents  = d.agent_outputs    || {};
  const prof    = agents.professor_agent || {};
  const retail  = agents.retail_agent    || {};
  const reasons = d.reason_stack     || [];
  const ai      = d.ai_overview?.overview || '';

  const lines = [
    '╭──────────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 7 — ML COUNCIL               │',
    '╰──────────────────────────────────────────────────────────────╯',
    '',
    '  Metric                           Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Council Bias                     ${council.council_bias || '—'}`,
    `  Tradeability Score               ${council.tradeability_score?.toFixed(3) || '—'}`,
    `  Alignment Class                  ${dis.alignment_class || '—'}`,
    `  Agreement Score                  ${dis.agreement_score?.toFixed(3) || '—'}`,
    `  Conflict Score                   ${dis.conflict_score?.toFixed(3) || '—'}`,
    '',
    '  Agent Outputs',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Professor    ${prof.predicted_label || '—'} (cal=${prof.calibrated_probability?.toFixed(3) || '—'})`,
    `  Retail       ${retail.predicted_label || '—'} (cal=${retail.calibrated_probability?.toFixed(3) || '—'})`,
  ];
  if (reasons.length) {
    lines.push('', '  Reason Stack:');
    reasons.forEach(r => lines.push(`  → ${r}`));
  }
  if (ai) {
    lines.push('', '  AI Council Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

export function buildP8Lines(d) {
  if (!d) return ['  Loading P8 — Decision & Risk...'];
  const exec  = d.execution_plan || {};
  const align = d.alignment      || {};
  const comps = align.components || [];
  const warns = d.warnings       || [];
  const ai    = d.ai_overview?.overview || '';

  const lines = [
    '╭────────────────────────────────────────────────────────────────╮',
    '│ BTC/USDT TERMINAL | LIVE PILLAR 8 — DECISION, RISK & SIZING    │',
    '╰────────────────────────────────────────────────────────────────╯',
    '',
    '  Metric                           Value',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Final Action                     ${d.final_action || '—'}`,
    `  Direction                        ${d.direction || '—'}`,
    `  Archetype                        ${d.decision_archetype || '—'}`,
    `  Confidence                       ${d.decision_confidence?.toFixed(3) || '—'}`,
    `  Tradability                      ${d.tradability_score?.toFixed(3) || '—'}`,
    `  Risk Score                       ${d.risk_score?.toFixed(3) || '—'} (${d.risk_state || '—'})`,
    `  Size Fraction                    ${d.size_fraction?.toFixed(4) || '—'}`,
    `  Max Leverage                     ${d.max_leverage_allowed?.toFixed(2) || '—'}×`,
    '',
    '  Execution Plan',
    '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `  Entry Style      ${exec.entry_style?.replace(/_/g,' ') || '—'}`,
    `  Stop Framework   ${exec.stop_framework?.replace(/_/g,' ') || '—'}`,
    `  Target Framework ${exec.target_framework?.replace(/_/g,' ') || '—'}`,
    `  Time Stop        ${exec.time_stop_bars ? exec.time_stop_bars+' bars' : '—'}`,
  ];
  if (comps.length) {
    lines.push('', '  Alignment per Pillar', '  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    comps.forEach(c => {
      const bias = c.raw_bias_score >= 0 ? '+'+c.raw_bias_score.toFixed(2) : c.raw_bias_score.toFixed(2);
      const wt   = c.weighted_score  >= 0 ? '+'+c.weighted_score.toFixed(3)  : c.weighted_score.toFixed(3);
      lines.push(`  ${(c.pillar || '').padEnd(14)} bias=${bias}  weighted=${wt}`);
    });
  }
  if (warns.length) {
    lines.push('', '  Warnings:');
    warns.forEach(w => lines.push(`  ⚠  ${w}`));
  }
  if ((exec.invalidators || []).length) {
    lines.push('', '  Invalidators:');
    exec.invalidators.forEach(inv => lines.push(`  —  ${inv}`));
  }
  if (ai) {
    lines.push('', '  AI Decision Overview:');
    ai.split('\n').filter(l => l.trim()).forEach(l => lines.push(`  ${l.trim()}`));
  }
  lines.push('', '  ' + '—'.repeat(56));
  return lines;
}

// ── Data extractors ───────────────────────────────────────────────────────────

function getSentimentState(p1) {
  if (!p1) return '—';
  return (p1.aggregate_sentiment?.label || 'NEUTRAL').toUpperCase();
}
function getSentimentColor(p1) {
  if (!p1) return 'dim';
  const label = (p1.aggregate_sentiment?.label || '').toUpperCase();
  return label === 'POSITIVE' ? 'g' : label === 'NEGATIVE' ? 'r' : 'dim';
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
  return (p4.candle_summary?.dominant_intent || 'BUY ABS').replace('_CANDIDATE','');
}
function getRegimeState(p5) {
  if (!p5) return '—';
  return (p5.regime_summary?.directional_regime || 'STRONG UP').replace(/_/g,' ');
}
function getEventState(p6) {
  if (!p6) return '—';
  return (p6.event || p6.event_name || 'EVENT').slice(0,8).toUpperCase();
}
function getCouncilState(p7) {
  if (!p7) return '—';
  return (p7.council?.council_bias || 'NO TRD').replace('_','-');
}
function getActionColor(action) {
  if (!action || action === '—') return 'var(--text)';
  if (action.includes('LONG'))    return 'var(--green)';
  if (action.includes('SHORT'))   return 'var(--red)';
  if (action === 'WATCHLIST')     return 'var(--yellow)';
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
  if (v == null)  return 'var(--text)';
  if (v >= 0.65)  return 'var(--green)';
  if (v >= 0.45)  return 'var(--yellow)';
  return 'var(--red)';
}