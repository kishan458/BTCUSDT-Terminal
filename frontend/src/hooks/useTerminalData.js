import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';
const WS_URL   = 'ws://localhost:8000/ws';

const INITIAL_STATE = {
  pillar1: null,
  pillar2: null,
  pillar3: null,
  pillar4: null,
  pillar5: null,
  pillar6: null,
  pillar7: null,
  pillar8: null,
};

export function useTerminalData() {
  const [data, setData]           = useState(INITIAL_STATE);
  const [lastUpdated, setLastUpdated] = useState({});
  const [connected, setConnected] = useState(false);
  const [loading, setLoading]     = useState(true);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  // ── Update a single pillar in state ────────────────────────────────────────
  const updatePillar = useCallback((pillar, pillarData) => {
    if (!pillarData) return;
    setData(prev => ({ ...prev, [pillar]: pillarData }));
    setLastUpdated(prev => ({ ...prev, [pillar]: new Date().toISOString() }));
  }, []);

  // ── Load initial snapshot via REST ─────────────────────────────────────────
  const loadSnapshot = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/snapshot`);
      const snap = res.data;
      setData({
        pillar1: snap.pillar1 || null,
        pillar2: snap.pillar2 || null,
        pillar3: snap.pillar3 || null,
        pillar4: snap.pillar4 || null,
        pillar5: snap.pillar5 || null,
        pillar6: snap.pillar6 || null,
        pillar7: snap.pillar7 || null,
        pillar8: snap.pillar8 || null,
      });
      setLoading(false);
    } catch (err) {
      console.error('[Terminal] Failed to load snapshot:', err.message);
      setLoading(false);
    }
  }, []);

  // ── WebSocket connection ────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log('[Terminal] WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'snapshot') {
          // Full snapshot on connect
          ['pillar1','pillar2','pillar3','pillar4','pillar5','pillar6','pillar7','pillar8'].forEach(p => {
            if (msg[p]) updatePillar(p, msg[p]);
          });
          setLoading(false);
        }

        if (msg.type === 'pillar_update') {
          // Single pillar update from scheduler
          updatePillar(msg.pillar, msg.data);
        }

      } catch (err) {
        console.error('[Terminal] WS message error:', err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('[Terminal] WebSocket disconnected — reconnecting in 3s...');
      reconnectTimer.current = setTimeout(connectWS, 3000);
    };

    ws.onerror = (err) => {
      console.error('[Terminal] WebSocket error:', err);
      ws.close();
    };
  }, [updatePillar]);

  // ── Manual refresh for a single pillar ─────────────────────────────────────
  const refreshPillar = useCallback(async (pillarNum) => {
    try {
      const res = await axios.get(`${API_BASE}/pillar${pillarNum}/refresh`);
      if (res.data?.data) {
        updatePillar(`pillar${pillarNum}`, res.data.data);
      }
      return res.data;
    } catch (err) {
      console.error(`[Terminal] Failed to refresh pillar${pillarNum}:`, err.message);
    }
  }, [updatePillar]);

  // ── Refresh all pillars ─────────────────────────────────────────────────────
  const refreshAll = useCallback(async () => {
    await loadSnapshot();
  }, [loadSnapshot]);

  // ── Sentiment history ───────────────────────────────────────────────────────
  const [sentimentHistory, setSentimentHistory] = useState([]);

  const loadSentimentHistory = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/pillar1/sentiment-history`);
      if (res.data?.history) {
        setSentimentHistory(res.data.history);
      }
    } catch (err) {
      console.error('[Terminal] Failed to load sentiment history:', err.message);
    }
  }, []);

  // ── Init ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    loadSnapshot();
    loadSentimentHistory();
    connectWS();

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [loadSnapshot, loadSentimentHistory, connectWS]);

  // Refresh sentiment history every 5 min
  useEffect(() => {
    const interval = setInterval(loadSentimentHistory, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadSentimentHistory]);

  return {
    data,
    lastUpdated,
    connected,
    loading,
    sentimentHistory,
    refreshPillar,
    refreshAll,
  };
}