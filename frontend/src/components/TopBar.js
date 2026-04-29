import { useState, useEffect, useRef } from 'react';

const styles = {
  topbar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 18px',
    height: '42px',
    background: 'var(--bg2)',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  left: { display: 'flex', alignItems: 'center', gap: '20px' },
  logo: {
    fontSize: '10px',
    fontWeight: 600,
    letterSpacing: '0.18em',
    color: 'var(--text3)',
    textTransform: 'uppercase',
  },
  priceBlock: { display: 'flex', alignItems: 'baseline', gap: '8px' },
  hl: { fontSize: '10px', color: 'var(--text3)' },
  hlVal: { color: 'var(--text2)' },
  right: { display: 'flex', alignItems: 'center', gap: '18px' },
  meta: { display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'var(--text2)' },
  dot: { width: '6px', height: '6px', borderRadius: '50%' },
  chip: {
    background: 'var(--yellow-dim)',
    border: '1px solid var(--yellow)',
    color: 'var(--yellow)',
    fontSize: '9px',
    fontWeight: 600,
    padding: '2px 7px',
    borderRadius: '2px',
    letterSpacing: '0.06em',
  },
  btn: {
    background: 'none',
    border: '1px solid var(--border2)',
    color: 'var(--text2)',
    fontFamily: 'var(--mono)',
    fontSize: '10px',
    padding: '3px 9px',
    cursor: 'pointer',
    borderRadius: '2px',
  },
};

export default function TopBar({ connected, onRefresh, pillar6Data }) {
  const [price, setPrice]         = useState(null);
  const [change, setChange]       = useState(null);
  const [high, setHigh]           = useState(null);
  const [low, setLow]             = useState(null);
  const [direction, setDirection] = useState('flat');
  const [flash, setFlash]         = useState('');
  const [lastUpdate, setLastUpdate] = useState('--');
  const lastPriceRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@miniTicker');
      wsRef.current = ws;
      ws.onmessage = (e) => {
        const d   = JSON.parse(e.data);
        const np  = parseFloat(d.c);
        const op  = parseFloat(d.o);
        const h   = parseFloat(d.h);
        const l   = parseFloat(d.l);
        const pct = ((np - op) / op) * 100;
        if (lastPriceRef.current !== null) {
          if (np > lastPriceRef.current) { setDirection('up');   setFlash('flash-up'); }
          else if (np < lastPriceRef.current) { setDirection('down'); setFlash('flash-down'); }
          setTimeout(() => setFlash(''), 350);
        }
        lastPriceRef.current = np;
        setPrice(np); setChange(pct); setHigh(h); setLow(l);
        setLastUpdate(new Date().toISOString().slice(11, 19) + ' UTC');
      };
      ws.onclose = () => setTimeout(connect, 3000);
      ws.onerror = () => ws.close();
    }
    connect();
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const eventLabel = (() => {
    if (!pillar6Data) return '⚡ FOMC IN 2D';
    const name = pillar6Data.event || pillar6Data.event_name;
    if (!name) return '⚡ FOMC IN 2D';
    return '⚡ ' + name.slice(0, 14).toUpperCase();
  })();

  const priceColor = direction === 'up' ? 'var(--green)' : direction === 'down' ? 'var(--red)' : 'var(--white)';

  return (
    <div style={styles.topbar}>
      <div style={styles.left}>
        <div style={styles.logo}>BTC/USDT ◆</div>
        <div style={styles.priceBlock}>
          <div className={flash} style={{ fontSize:'19px', fontWeight:600, letterSpacing:'-0.02em', color:priceColor, transition:'color 0.25s', borderRadius:'2px', padding:'0 2px' }}>
            {price ? '$' + price.toLocaleString('en-US', { minimumFractionDigits:2, maximumFractionDigits:2 }) : '—'}
          </div>
          {change !== null && (
            <div style={{ fontSize:'11px', fontWeight:500, color: change >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}% (24h)
            </div>
          )}
        </div>
        <div style={styles.hl}>
          H <span style={styles.hlVal}>{high ? '$' + Math.round(high).toLocaleString() : '—'}</span>
          &nbsp;&nbsp;
          L <span style={styles.hlVal}>{low ? '$' + Math.round(low).toLocaleString() : '—'}</span>
        </div>
      </div>

      <div style={styles.right}>
        <div style={styles.meta}>
          <div style={{ ...styles.dot, background:'var(--green)' }} />
          <span>{getSession()}</span>
        </div>
        <div style={styles.chip}>{eventLabel}</div>
        <div style={styles.meta}>
          <div style={{ ...styles.dot, background: connected ? 'var(--green)' : 'var(--yellow)' }} />
          <span>Live · {lastUpdate}</span>
        </div>
        <button style={styles.btn} onClick={onRefresh}>↺ Refresh All</button>
      </div>
    </div>
  );
}

function getSession() {
  const h = new Date().getUTCHours();
  if (h >= 0  && h < 8)  return 'ASIA SESSION';
  if (h >= 8  && h < 16) return 'LONDON SESSION';
  return 'NY SESSION';
}