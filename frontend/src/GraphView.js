import React, { useRef, useEffect, useState, useCallback } from 'react';

const API_BASE = '/api';

/* エンティティタイプ設定 */
const TYPE_CONFIG = {
  person:       { color: '#ef4444', icon: '👤', glow: 'rgba(239,68,68,0.25)' },
  organization: { color: '#2dd4bf', icon: '🏛', glow: 'rgba(45,212,191,0.25)' },
  technology:   { color: '#a78bfa', icon: '⚡', glow: 'rgba(167,139,250,0.25)' },
  location:     { color: '#4ade80', icon: '📍', glow: 'rgba(74,222,128,0.25)' },
  event:        { color: '#f472b6', icon: '📅', glow: 'rgba(244,114,182,0.25)' },
  concept:      { color: '#60a5fa', icon: '💡', glow: 'rgba(96,165,250,0.25)' },
  date:         { color: '#fb923c', icon: '📌', glow: 'rgba(251,146,60,0.25)' },
};

export default function GraphView() {
  const canvasRef = useRef(null);
  const [graphData, setGraphData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const stateRef = useRef({
    nodes: [], links: [], nodeMap: {},
    transform: { x: 0, y: 0, k: 1 },
    hoveredNode: null, dragNode: null,
    panning: false, panStart: { x: 0, y: 0 },
    alpha: 1, animId: null, selectedId: null,
  });

  /* データ取得（クエリ付き） */
  const fetchGraph = useCallback(async (query = '') => {
    try {
      setSearching(!!query);
      const url = query
        ? `${API_BASE}/graph/data?query=${encodeURIComponent(query)}`
        : `${API_BASE}/graph/data`;
      const res = await fetch(url);
      if (res.ok) { setGraphData(await res.json()); setError(''); }
      else setError('グラフデータの取得に失敗しました');
    } catch (e) { setError(`接続エラー: ${e.message}`); }
    finally { setSearching(false); }
  }, []);

  useEffect(() => { fetchGraph(); }, [fetchGraph]);

  /* メイン: Canvas力学シミュレーション */
  useEffect(() => {
    if (!graphData || !canvasRef.current || graphData.nodes.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const S = stateRef.current;

    /* ノード・リンク初期化 */
    S.nodes = graphData.nodes.map(n => ({ ...n, x: 0, y: 0, vx: 0, vy: 0, fx: null, fy: null }));
    S.links = graphData.edges.map(e => ({ ...e }));
    S.nodeMap = {};
    S.nodes.forEach(n => { S.nodeMap[n.id] = n; });
    S.links.forEach(l => {
      l.sourceNode = S.nodeMap[l.source];
      l.targetNode = S.nodeMap[l.target];
    });
    S.alpha = 1;

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
    }
    resize();
    const onResize = () => resize();
    window.addEventListener('resize', onResize);

    const W = () => canvas.width / dpr;
    const H = () => canvas.height / dpr;

    /* 初期配置: 円形 */
    const cx0 = W() / 2, cy0 = H() / 2;
    const radius = Math.min(W(), H()) * 0.3;
    S.nodes.forEach((n, i) => {
      const angle = (i / S.nodes.length) * Math.PI * 2;
      n.x = cx0 + Math.cos(angle) * radius + (Math.random() - 0.5) * 30;
      n.y = cy0 + Math.sin(angle) * radius + (Math.random() - 0.5) * 30;
    });

    /* 座標変換 */
    const T = S.transform;
    const toWorld = (px, py) => ({
      x: (px - T.x) / T.k,
      y: (py - T.y) / T.k,
    });

    const nodeR = (n) => Math.max(20, Math.min(34, 16 + n.degree * 3));

    const nodeAt = (px, py) => {
      const w = toWorld(px, py);
      for (let i = S.nodes.length - 1; i >= 0; i--) {
        const n = S.nodes[i];
        const r = nodeR(n);
        const dx = w.x - n.x, dy = w.y - n.y;
        if (dx * dx + dy * dy < (r + 4) * (r + 4)) return n;
      }
      return null;
    };

    /* マウスイベント */
    const onMouseDown = (e) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const n = nodeAt(px, py);
      if (n) { S.dragNode = n; n.fx = n.x; n.fy = n.y; }
      else { S.panning = true; S.panStart = { x: e.clientX - T.x, y: e.clientY - T.y }; }
    };

    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      if (S.dragNode) {
        const w = toWorld(px, py);
        S.dragNode.fx = w.x; S.dragNode.fy = w.y;
        S.dragNode.x = w.x; S.dragNode.y = w.y;
        S.alpha = Math.max(S.alpha, 0.3);
      } else if (S.panning) {
        T.x = e.clientX - S.panStart.x;
        T.y = e.clientY - S.panStart.y;
      } else {
        S.hoveredNode = nodeAt(px, py);
        canvas.style.cursor = S.hoveredNode ? 'pointer' : 'grab';
      }
    };

    const onMouseUp = () => {
      if (S.dragNode) { S.dragNode.fx = null; S.dragNode.fy = null; S.dragNode = null; }
      S.panning = false;
    };

    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const n = nodeAt(e.clientX - rect.left, e.clientY - rect.top);
      S.selectedId = n ? n.id : null;
      setSelectedNode(n || null);
    };

    const onWheel = (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const newK = Math.max(0.15, Math.min(5, T.k * factor));
      T.x = mx - (mx - T.x) * (newK / T.k);
      T.y = my - (my - T.y) * (newK / T.k);
      T.k = newK;
    };

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('mouseleave', onMouseUp);
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    /* 描画 */
    function draw() {
      const w = W(), h = H();
      ctx.save();
      ctx.scale(dpr, dpr);

      /* 背景 */
      ctx.fillStyle = '#0f1117';
      ctx.fillRect(0, 0, w, h);

      ctx.save();
      ctx.translate(T.x, T.y);
      ctx.scale(T.k, T.k);

      /* 検索中かどうか（非ハイライトノードを暗くする） */
      const hasSearch = S.nodes.some(n => n.highlighted);

      /* エッジ描画 */
      S.links.forEach(l => {
        const s = l.sourceNode, t = l.targetNode;
        if (!s || !t) return;
        const isHL = l.highlighted && hasSearch;

        /* ライン */
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.strokeStyle = isHL ? 'rgba(250,204,21,0.6)' :
          (hasSearch ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.1)');
        ctx.lineWidth = isHL ? 2 : 1.2;
        ctx.stroke();

        /* 矢印ヘッド */
        const tr = nodeR(t);
        const dx = t.x - s.x, dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const ux = dx / dist, uy = dy / dist;
        const ax = t.x - ux * (tr + 4), ay = t.y - uy * (tr + 4);
        const aSize = 6;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - ux * aSize - uy * aSize * 0.5, ay - uy * aSize + ux * aSize * 0.5);
        ctx.lineTo(ax - ux * aSize + uy * aSize * 0.5, ay - uy * aSize - ux * aSize * 0.5);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.fill();

        /* エッジラベル */
        if (l.relation && T.k > 0.4) {
          const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
          const angle = Math.atan2(dy, dx);
          ctx.save();
          ctx.translate(mx, my);
          let drawAngle = angle;
          if (angle > Math.PI / 2 || angle < -Math.PI / 2) drawAngle += Math.PI;
          ctx.rotate(drawAngle);
          ctx.font = '11px "Meiryo", "Segoe UI", sans-serif';
          ctx.fillStyle = 'rgba(255,255,255,0.5)';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(l.relation, 0, -5);
          ctx.restore();
        }
      });

      /* ノード描画 */
      S.nodes.forEach(n => {
        const cfg = TYPE_CONFIG[n.type] || TYPE_CONFIG.concept;
        const r = nodeR(n);
        const isHovered = S.hoveredNode === n;
        const isSelected = S.selectedId === n.id;
        const isHighlighted = !!n.highlighted;
        const dimmed = hasSearch && !isHighlighted && !isHovered && !isSelected;

        /* 検索ハイライト（脈動グロー） */
        if (isHighlighted) {
          const pulse = 0.6 + 0.4 * Math.sin(Date.now() / 300);
          const glowR = r + 25 * pulse;
          const grd = ctx.createRadialGradient(n.x, n.y, r * 0.5, n.x, n.y, glowR);
          grd.addColorStop(0, cfg.color + 'aa');
          grd.addColorStop(0.5, cfg.color + '44');
          grd.addColorStop(1, 'transparent');
          ctx.beginPath();
          ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
          ctx.fillStyle = grd;
          ctx.fill();
        }

        /* グロー（ホバー/選択時） */
        if ((isHovered || isSelected) && !isHighlighted) {
          const grd = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, r + 20);
          grd.addColorStop(0, cfg.glow);
          grd.addColorStop(1, 'transparent');
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 20, 0, Math.PI * 2);
          ctx.fillStyle = grd;
          ctx.fill();
        }

        /* 外側リング */
        ctx.globalAlpha = dimmed ? 0.15 : 1;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = cfg.color + '18';
        ctx.fill();

        /* 本体（グラデーション） */
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(
          n.x - r * 0.25, n.y - r * 0.25, r * 0.1,
          n.x, n.y, r
        );
        grad.addColorStop(0, cfg.color + 'bb');
        grad.addColorStop(0.7, cfg.color + '77');
        grad.addColorStop(1, cfg.color + '44');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = cfg.color + (isHighlighted ? 'ff' : (isHovered || isSelected ? 'ff' : '99'));
        ctx.lineWidth = isHighlighted ? 3 : (isHovered || isSelected ? 2.5 : 1);
        ctx.stroke();

        /* アイコン */
        ctx.font = `${r * 0.65}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(cfg.icon, n.x, n.y + 1);

        /* ラベル（ノード名） */
        ctx.font = `bold 13px "Meiryo", "Segoe UI", sans-serif`;
        ctx.fillStyle = dimmed ? 'rgba(255,255,255,0.15)' : '#ffffffdd';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(n.label, n.x, n.y + r + 8);

        /* スコアバッジ（検索ヒット時） */
        if (isHighlighted && n.search_score > 0) {
          ctx.font = 'bold 10px monospace';
          ctx.fillStyle = '#facc15';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText((n.search_score * 100).toFixed(0) + '%', n.x, n.y - r - 6);
        }

        ctx.globalAlpha = 1;
      });

      ctx.restore();
      ctx.restore();
    }

    /* 力学シミュレーション */
    function tick() {
      const nodes = S.nodes, links = S.links;

      /* 斥力 */
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = b.x - a.x || 0.1, dy = b.y - a.y || 0.1;
          let dist2 = dx * dx + dy * dy;
          if (dist2 < 1) dist2 = 1;
          let dist = Math.sqrt(dist2);
          let force = -600 / dist2;
          a.vx += (dx / dist) * force;
          a.vy += (dy / dist) * force;
          b.vx -= (dx / dist) * force;
          b.vy -= (dy / dist) * force;
        }
      }

      /* 引力 */
      links.forEach(l => {
        const s = l.sourceNode, t = l.targetNode;
        if (!s || !t) return;
        let dx = t.x - s.x, dy = t.y - s.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = (dist - 140) * 0.004;
        s.vx += (dx / dist) * force;
        s.vy += (dy / dist) * force;
        t.vx -= (dx / dist) * force;
        t.vy -= (dy / dist) * force;
      });

      /* 中心引力 */
      const cxc = W() / 2, cyc = H() / 2;
      nodes.forEach(n => {
        n.vx += (cxc - n.x) * 0.0003;
        n.vy += (cyc - n.y) * 0.0003;
      });

      /* 更新 */
      nodes.forEach(n => {
        if (n.fx != null) { n.x = n.fx; n.vx = 0; }
        else { n.vx *= 0.82; n.x += n.vx * S.alpha; }
        if (n.fy != null) { n.y = n.fy; n.vy = 0; }
        else { n.vy *= 0.82; n.y += n.vy * S.alpha; }
      });

      S.alpha = Math.max(0.001, S.alpha * 0.995);
      draw();
      S.animId = requestAnimationFrame(tick);
    }

    tick();

    return () => {
      cancelAnimationFrame(S.animId);
      window.removeEventListener('resize', onResize);
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseup', onMouseUp);
      canvas.removeEventListener('mouseleave', onMouseUp);
      canvas.removeEventListener('click', onClick);
      canvas.removeEventListener('wheel', onWheel);
    };
  }, [graphData]);

  /* --- Render --- */

  if (error) {
    return <div style={css.center}><span style={{ color: '#ef4444' }}>{error}</span></div>;
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div style={css.center}>
        <p style={{ fontSize: 16, color: '#888', marginBottom: 8 }}>グラフが空です</p>
        <p style={{ fontSize: 13, color: '#555' }}>サイドバーからドキュメントを追加してください</p>
      </div>
    );
  }

  const stats = graphData.stats || {};
  const typeCounts = graphData.type_counts || {};

  return (
    <div style={css.wrapper}>
      {/* Canvas領域 */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <canvas ref={canvasRef} />

        {/* 凡例 */}
        <div style={css.legend}>
          {Object.entries(typeCounts).map(([type, count]) => {
            const cfg = TYPE_CONFIG[type] || TYPE_CONFIG.concept;
            return (
              <span key={type} style={css.legendItem}>
                <span style={{ ...css.dot, background: cfg.color }} />
                {type} ({count})
              </span>
            );
          })}
          <span style={{ color: '#555', marginLeft: 8, fontSize: 11 }}>
            {graphData.nodes.length} nodes・{graphData.edges.length} edges
          </span>
        </div>

        {/* 検索バー + ボタン */}
        <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') fetchGraph(searchQuery);
            }}
            placeholder="ベクトル検索..."
            style={{
              width: 200, padding: '6px 12px', fontSize: 12,
              background: 'rgba(255,255,255,0.06)', color: '#eee',
              border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6,
              outline: 'none',
            }}
          />
          <button onClick={() => fetchGraph(searchQuery)}
            style={{ ...css.btn, background: searching ? 'rgba(37,99,235,0.3)' : css.btn.background }}>
            {searching ? '検索中...' : '🔍 検索'}
          </button>
          {searchQuery && (
            <button onClick={() => { setSearchQuery(''); fetchGraph(''); }}
              style={css.btn}>
              ✕ クリア
            </button>
          )}
          <button onClick={() => fetchGraph('')} style={css.btn}>更新</button>
        </div>

        {/* 検索結果サマリー */}
        {graphData.nodes.some(n => n.highlighted) && (
          <div style={{
            position: 'absolute', top: 50, right: 12,
            padding: '6px 14px', background: 'rgba(250,204,21,0.1)',
            border: '1px solid rgba(250,204,21,0.3)', borderRadius: 6,
            color: '#facc15', fontSize: 12,
          }}>
            🔍 {graphData.nodes.filter(n => n.highlighted).length}件のエンティティがヒット
          </div>
        )}

        {/* 選択ノード詳細 */}
        {selectedNode && (
          <div style={css.detail}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                ...css.dot, width: 12, height: 12,
                background: (TYPE_CONFIG[selectedNode.type] || TYPE_CONFIG.concept).color,
              }} />
              <span style={{ fontWeight: 700, fontSize: 15, color: '#fff' }}>
                {selectedNode.label}
              </span>
              <span style={css.badge}>{selectedNode.type}</span>
              <span style={{ fontSize: 11, color: '#777' }}>
                接続数: {selectedNode.degree}
              </span>
              <button onClick={() => { setSelectedNode(null); stateRef.current.selectedId = null; }}
                style={{ marginLeft: 'auto', background: 'none', border: 'none',
                  color: '#555', cursor: 'pointer', fontSize: 20, lineHeight: 1 }}>×</button>
            </div>
          </div>
        )}
      </div>

      {/* 統計パネル */}
      <div style={css.statsPanel}>
        <h3 style={css.statsTitle}>📊 グラフ統計（NetworkX）</h3>

        <div style={css.grid}>
          <div style={css.card}>
            <div style={css.num}>{stats.nodes ?? graphData.nodes.length}</div>
            <div style={css.label}>ノード数</div>
          </div>
          <div style={css.card}>
            <div style={css.num}>{stats.edges ?? graphData.edges.length}</div>
            <div style={css.label}>エッジ数</div>
          </div>
          <div style={css.card}>
            <div style={{ ...css.num, fontSize: 24 }}>
              {(stats.density ?? 0).toFixed(3)}
            </div>
            <div style={css.label}>密度</div>
          </div>
          <div style={css.card}>
            <div style={css.num}>{stats.components ?? 0}</div>
            <div style={css.label}>連結成分</div>
          </div>
        </div>

        {stats.centrality_top && stats.centrality_top.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize: 12, color: '#aaa', marginBottom: 8 }}>
              🏆 中心性TOP:
            </div>
            <div style={{ fontSize: 12, color: '#ccc', lineHeight: 1.8 }}>
              {stats.centrality_top.map((c, i) => (
                <span key={i}>
                  {c.name}({c.score.toFixed(3)})
                  {i < stats.centrality_top.length - 1 ? ', ' : ''}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ベクトル検索ヒット */}
        {graphData.nodes.filter(n => n.highlighted).length > 0 && (
          <div style={{ marginTop: 20, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16 }}>
            <div style={{ fontSize: 12, color: '#facc15', marginBottom: 8 }}>
              🔍 ベクトル検索ヒット:
            </div>
            {graphData.nodes
              .filter(n => n.highlighted)
              .sort((a, b) => b.search_score - a.search_score)
              .map((n, i) => (
              <div key={i} style={{
                fontSize: 12, color: '#ccc', marginBottom: 4,
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <span>
                  <span style={{
                    ...css.dot, width: 8, height: 8, marginRight: 6,
                    background: (TYPE_CONFIG[n.type] || TYPE_CONFIG.concept).color,
                  }} />
                  {n.label}
                </span>
                <span style={{ fontSize: 10, color: '#666', fontFamily: 'monospace' }}>
                  {(n.search_score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* スタイル定数 */
const css = {
  wrapper: { display: 'flex', width: '100%', height: '100%', background: '#0f1117' },
  center: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', height: '100%', background: '#0f1117',
  },
  legend: {
    position: 'absolute', top: 12, left: 12, display: 'flex',
    alignItems: 'center', gap: 12, background: 'rgba(15,17,23,0.88)',
    border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8,
    padding: '8px 14px', fontSize: 12, color: '#bbb', flexWrap: 'wrap',
  },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5 },
  dot: { width: 10, height: 10, borderRadius: '50%', display: 'inline-block' },
  btn: {
    padding: '6px 14px', background: 'rgba(255,255,255,0.06)',
    border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6,
    color: '#aaa', cursor: 'pointer', fontSize: 12,
  },
  detail: {
    position: 'absolute', bottom: 14, left: 14, right: 14,
    background: 'rgba(15,17,23,0.92)', border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 10, padding: '10px 14px',
  },
  badge: {
    fontSize: 10, background: 'rgba(255,255,255,0.08)', color: '#999',
    padding: '2px 8px', borderRadius: 10,
  },
  statsPanel: {
    width: 280, background: '#141620', borderLeft: '1px solid rgba(255,255,255,0.06)',
    padding: 20, overflowY: 'auto', flexShrink: 0,
  },
  statsTitle: { fontSize: 14, fontWeight: 600, color: '#2dd4bf', margin: '0 0 16px' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
  card: {
    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 8, padding: '14px 12px',
  },
  num: { fontSize: 30, fontWeight: 700, color: '#2dd4bf', fontFamily: 'monospace' },
  label: { fontSize: 11, color: '#777', marginTop: 2 },
};