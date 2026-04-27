import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import GraphView from './GraphView';

const API_BASE = '/api';

/* ============================================================
   スタイル
   ============================================================ */
const styles = {
  app: {
    display: 'flex', height: '100vh', fontFamily: '"Segoe UI", "Meiryo", sans-serif',
    background: '#f5f5f5', color: '#1a1a1a',
  },
  sidebar: {
    width: 280, background: '#fff', borderRight: '1px solid #e0e0e0',
    display: 'flex', flexDirection: 'column', padding: 16,
  },
  main: {
    flex: 1, display: 'flex', flexDirection: 'column',
  },
  header: {
    padding: '16px 24px', background: '#fff', borderBottom: '1px solid #e0e0e0',
    display: 'flex', alignItems: 'center', gap: 12,
  },
  chatArea: {
    flex: 1, overflowY: 'auto', padding: '16px 24px',
  },
  inputBar: {
    padding: '12px 24px', background: '#fff', borderTop: '1px solid #e0e0e0',
    display: 'flex', gap: 8,
  },
  input: {
    flex: 1, padding: '10px 16px', border: '1px solid #d0d0d0',
    borderRadius: 8, fontSize: 14, outline: 'none',
  },
  sendBtn: {
    padding: '10px 20px', background: '#2563eb', color: '#fff',
    border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14,
    fontWeight: 500,
  },
  msgUser: {
    maxWidth: '70%', marginLeft: 'auto', marginBottom: 12,
    padding: '10px 16px', background: '#2563eb', color: '#fff',
    borderRadius: '16px 16px 4px 16px', lineHeight: 1.5,
  },
  msgBot: {
    maxWidth: '80%', marginBottom: 12,
    padding: '12px 16px', background: '#fff', border: '1px solid #e0e0e0',
    borderRadius: '16px 16px 16px 4px', lineHeight: 1.6,
  },
  nodeIndicator: {
    fontSize: 11, color: '#888', marginBottom: 4,
    display: 'flex', alignItems: 'center', gap: 6,
  },
  statsCard: {
    background: '#f8f9fa', borderRadius: 8, padding: 12, marginBottom: 12,
    border: '1px solid #e8e8e8',
  },
  statLabel: { fontSize: 11, color: '#888', marginBottom: 2 },
  statValue: { fontSize: 20, fontWeight: 600, color: '#1a1a1a' },
};

/* ============================================================
   メインApp
   ============================================================ */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeNode, setActiveNode] = useState('');
  const [graphStats, setGraphStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [docText, setDocText] = useState('');
  const [docSource, setDocSource] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [view, setView] = useState('chat'); // 'chat' or 'graph'
  const chatEndRef = useRef(null);

  // スクロール
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeNode]);

  // 初期データ取得
  const fetchStats = useCallback(async () => {
    try {
      const [statsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/graph/stats`),
        fetch(`${API_BASE}/health`),
      ]);
      if (statsRes.ok) setGraphStats(await statsRes.json());
      if (healthRes.ok) setHealth(await healthRes.json());
    } catch (e) {
      console.warn('Stats fetch failed:', e);
    }
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  // チャット送信（SSEストリーミング）
  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setActiveNode('');

    // 履歴（直近6件）
    const history = messages.slice(-6).map(m => ({
      role: m.role, content: m.content,
    }));

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botAnswer = '';
      let meta = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

        for (const line of lines) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            setActiveNode(parsed.node);

            // 各ノードの出力を処理
            if (parsed.data.answer) {
              botAnswer = parsed.data.answer;
            }
            if (parsed.data.quality_score !== undefined) {
              meta.quality_score = parsed.data.quality_score;
            }
            if (parsed.data.retry_count !== undefined) {
              meta.retry_count = parsed.data.retry_count;
            }
            if (parsed.data.search_results) {
              meta.sources = parsed.data.search_results;
            }
            if (parsed.data.matched_entities) {
              meta.matched_entities = parsed.data.matched_entities;
            }
          } catch (e) {
            // パースエラーは無視
          }
        }
      }

      // 最終回答をメッセージに追加
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: botAnswer || '回答を生成できませんでした。',
        meta,
      }]);

      // グラフ統計を更新
      fetchStats();

    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `エラー: ${err.message}\nバックエンド (localhost:8000) が起動しているか確認してください。`,
      }]);
    }

    setLoading(false);
    setActiveNode('');
  };

  // ドキュメント投入
  const handleDocUpload = async () => {
    const text = docText.trim();
    if (!text || uploading) return;

    setUploading(true);
    setUploadMsg('');

    try {
      const res = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          metadata: { source: docSource || 'manual' },
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setUploadMsg(`追加完了 (ID: ${data.doc_id})`);
        setDocText('');
        setDocSource('');
        fetchStats();
      } else {
        const err = await res.json();
        setUploadMsg(`エラー: ${err.detail || res.statusText}`);
      }
    } catch (e) {
      setUploadMsg(`接続エラー: ${e.message}`);
    }

    setUploading(false);
    setTimeout(() => setUploadMsg(''), 5000);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* --- サイドバー --- */
  const Sidebar = () => (
    <div style={styles.sidebar}>
      <h2 style={{ margin: '0 0 16px', fontSize: 18 }}>GraphRAG LLM</h2>

      {/* 接続状態 */}
      <div style={styles.statsCard}>
        <div style={styles.statLabel}>Ollama</div>
        <div style={{ fontSize: 14, color: health?.ollama ? '#16a34a' : '#dc2626' }}>
          {health?.ollama ? '● 接続中' : '○ 未接続'}
        </div>
      </div>

      {/* グラフ統計 */}
      {graphStats && (
        <>
          <div style={styles.statsCard}>
            <div style={styles.statLabel}>ノード数</div>
            <div style={styles.statValue}>{graphStats.nodes}</div>
          </div>
          <div style={styles.statsCard}>
            <div style={styles.statLabel}>エッジ数</div>
            <div style={styles.statValue}>{graphStats.edges}</div>
          </div>
          <div style={styles.statsCard}>
            <div style={styles.statLabel}>コミュニティ</div>
            <div style={styles.statValue}>{graphStats.communities}</div>
          </div>
          <div style={styles.statsCard}>
            <div style={styles.statLabel}>チャンク数</div>
            <div style={styles.statValue}>{graphStats.chunks}</div>
          </div>
        </>
      )}

      {/* ドキュメント投入 */}
      <div style={{ marginTop: 16, borderTop: '1px solid #e8e8e8', paddingTop: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>ドキュメント追加</div>
        <textarea
          value={docText}
          onChange={e => setDocText(e.target.value)}
          placeholder="テキストを貼り付け..."
          style={{
            width: '100%', height: 100, padding: 8, fontSize: 12,
            border: '1px solid #d0d0d0', borderRadius: 6, resize: 'vertical',
            fontFamily: 'inherit', boxSizing: 'border-box',
          }}
        />
        <input
          value={docSource}
          onChange={e => setDocSource(e.target.value)}
          placeholder="ソース名（任意）"
          style={{
            width: '100%', padding: '6px 8px', fontSize: 12, marginTop: 6,
            border: '1px solid #d0d0d0', borderRadius: 6, boxSizing: 'border-box',
          }}
        />
        <button
          onClick={handleDocUpload}
          disabled={uploading || !docText.trim()}
          style={{
            width: '100%', marginTop: 8, padding: '8px 0',
            background: uploading || !docText.trim() ? '#ccc' : '#16a34a',
            color: '#fff', border: 'none', borderRadius: 6,
            cursor: uploading || !docText.trim() ? 'default' : 'pointer',
            fontSize: 13, fontWeight: 500,
          }}
        >
          {uploading ? '追加中...' : 'グラフに追加'}
        </button>
        {uploadMsg && (
          <div style={{
            marginTop: 6, fontSize: 11, padding: '4px 8px', borderRadius: 4,
            background: uploadMsg.startsWith('エラー') || uploadMsg.startsWith('接続') ? '#fef2f2' : '#f0fdf4',
            color: uploadMsg.startsWith('エラー') || uploadMsg.startsWith('接続') ? '#dc2626' : '#16a34a',
          }}>
            {uploadMsg}
          </div>
        )}
      </div>

      <div style={{ marginTop: 'auto', fontSize: 12, color: '#999' }}>
        Model: qwen2.5:7b<br/>
        Embed: nomic-embed-text
      </div>
    </div>
  );

  /* --- ノード名の日本語表示 --- */
  const nodeLabel = (name) => {
    const map = {
      analyze_query: '🔍 クエリ分析中...',
      search: '📊 GraphRAG検索中...',
      generate: '🤖 回答生成中...',
      check_quality: '✅ 品質チェック中...',
    };
    return map[name] || name;
  };

  return (
    <div style={styles.app}>
      <Sidebar />
      <div style={styles.main}>
        {/* ヘッダー */}
        <div style={styles.header}>
          <div style={{ display: 'flex', gap: 4 }}>
            {['chat', 'graph'].map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                style={{
                  padding: '6px 16px', fontSize: 14, fontWeight: view === v ? 600 : 400,
                  background: view === v ? '#2563eb' : 'transparent',
                  color: view === v ? '#fff' : '#666',
                  border: view === v ? 'none' : '1px solid #d0d0d0',
                  borderRadius: 6, cursor: 'pointer',
                }}
              >
                {v === 'chat' ? 'チャット' : 'グラフ'}
              </button>
            ))}
          </div>
          {loading && activeNode && (
            <span style={styles.nodeIndicator}>
              {nodeLabel(activeNode)}
            </span>
          )}
        </div>

        {/* グラフビュー */}
        {view === 'graph' && (
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <GraphView />
          </div>
        )}

        {/* チャットエリア */}
        {view === 'chat' && (
        <>
        <div style={styles.chatArea}>
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: '#999', marginTop: 80 }}>
              <p style={{ fontSize: 18, marginBottom: 8 }}>GraphRAG Local LLM</p>
              <p style={{ fontSize: 13 }}>
                知識グラフを活用したローカルLLMアシスタント
              </p>
              <p style={{ fontSize: 12, color: '#bbb', marginTop: 16 }}>
                左のサイドバーからドキュメントを追加すると、<br/>
                より正確な回答が得られます
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i}>
              {msg.role === 'user' ? (
                <div style={styles.msgUser}>{msg.content}</div>
              ) : (
                <div>
                  <div style={styles.msgBot}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                    {msg.meta?.quality_score !== undefined && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#999',
                        borderTop: '1px solid #f0f0f0', paddingTop: 6,
                        display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span>
                          品質スコア: {msg.meta.quality_score.toFixed(2)}
                          {msg.meta.retry_count > 1 &&
                            ` (${msg.meta.retry_count - 1}回リトライ)`}
                          {msg.meta.sources?.length > 0 &&
                            ` | ソース: ${msg.meta.sources.length}件`}
                        </span>
                        {msg.meta.matched_entities?.length > 0 && (
                          <button
                            onClick={() => setView('graph')}
                            style={{
                              fontSize: 11, padding: '2px 8px', background: '#2563eb',
                              color: '#fff', border: 'none', borderRadius: 4,
                              cursor: 'pointer',
                            }}
                          >
                            グラフで表示 ({msg.meta.matched_entities.length}件)
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div style={styles.inputBar}>
          <input
            style={styles.input}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="質問を入力... (Enter で送信)"
            disabled={loading}
          />
          <button
            style={{
              ...styles.sendBtn,
              opacity: loading ? 0.6 : 1,
            }}
            onClick={handleSend}
            disabled={loading}
          >
            {loading ? '処理中...' : '送信'}
          </button>
        </div>
        </>
        )}
      </div>
    </div>
  );
}