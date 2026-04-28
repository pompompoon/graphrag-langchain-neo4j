# GraphRAG Local LLM

ローカルLLMと知識グラフを組み合わせたRAG（Retrieval-Augmented Generation）チャットアプリケーション。
テキストからエンティティを自動抽出し、知識グラフを構築・可視化・検索する。

<img width="2000" height="875" alt="image" src="https://github.com/user-attachments/assets/62302da8-1066-4905-8c10-0e0ad9a35644" />
<img width="2000" height="873" alt="image" src="https://github.com/user-attachments/assets/7eb78310-c9d8-46e0-9431-d6b02dcfcc2a" />


## アーキテクチャ

```
React (localhost:3000)     ← チャットUI + グラフ可視化
  ↓
Apache (localhost:80)      ← リバースプロキシ
  ↓
FastAPI (localhost:8000)   ← API サーバー
  ↓
LangGraph Agent            ← 自律ループ付きワークフロー
  ├─ クエリ分析ノード         LLMで検索クエリを最適化
  ├─ GraphRAG検索ノード       ベクトル検索 + グラフ近傍展開
  ├─ 回答生成ノード           LLMで回答生成
  └─ 品質チェックノード        LLM自己評価 → 品質NGならループ再検索
  ↓
config.py                  ← .env で全設定を一元管理
  ├─ PROVIDER: ollama / gemini
  └─ STORAGE:  file / neo4j
```

## 主な機能

- LLMによる単語単位のエンティティ抽出（人物・組織・技術・概念など7タイプ）
- 知識グラフの自動構築（エンティティ間のリレーション付き有向グラフ）
- ダークテーマのCanvas力学シミュレーションによるインタラクティブ可視化
- ベクトル検索によるグラフ上のエンティティハイライト（脈動グロー表示）
- LangGraphによる品質チェックループ付きRAGチャット（SSEストリーミング）
- LangChain抽象レイヤーによるLLM/Embeddingsのプロバイダ切り替え
- Neo4j / ファイルベースの2モードストレージ

## 技術スタック

| レイヤー | 技術 |
|---|---|
| フロントエンド | React 18, Canvas (力学シミュレーション) |
| リバースプロキシ | Apache (mod_proxy) |
| APIサーバー | FastAPI, Uvicorn |
| エージェント | LangGraph (LangChain) |
| LLM | Ollama (qwen2.5:7b) / Google Gemini (無料枠) |
| 埋め込み | nomic-embed-text / gemini-embedding-001 |
| ストレージ (Neo4j) | Neo4j 5.x (グラフ + ベクトルインデックス) |
| ストレージ (ファイル) | NetworkX + hnswlib + JSON |

## ディレクトリ構成

```
graphrag-app/
├── backend/
│   ├── .env                    # 環境設定
│   ├── config.py               # LLM/Embeddings/ストレージの一元管理
│   ├── main.py                 # FastAPI エンドポイント
│   ├── graph_agent.py          # LangGraph エージェント
│   ├── entity_extractor.py     # LLMエンティティ抽出
│   ├── neo4j_store.py          # Neo4jストレージバックエンド
│   ├── graph_rag.py            # ファイルベースストレージ
│   ├── requirements.txt
│   └── data/                   # ファイルモード時の永続化データ
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js
│       ├── App.js              # チャットUI + サイドバー
│       └── GraphView.js        # グラフ可視化
├── config/
│   └── httpd-graphrag.conf     # Apache リバースプロキシ設定
├── start.bat
├── README.md
└── Tuning guide.md
```

## セットアップ

### 前提条件

- Windows 10/11
- Anaconda (Python 3.11)
- Ollama
- Node.js（conda経由でもOK）
- Neo4j 5.11+（Neo4jモードの場合）

### 1. Ollama モデル準備

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. Python環境構築

```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
cd backend
pip install chroma-hnswlib
pip install fastapi "uvicorn[standard]" pydantic langchain langchain-core langchain-ollama langchain-google-genai langgraph networkx numpy python-dotenv httpx neo4j
```

### 3. Node.js 準備

```bash
conda install nodejs -y
cd frontend
npm install
```

### 4. ストレージ設定

#### Neo4j モード（推奨）

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
```

`backend/.env` を設定：

```env
STORAGE=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

#### ファイルモード（Neo4j不要）

```env
STORAGE=file
```

## 起動方法

ターミナルを2つ開いて実行：

**ターミナル1: バックエンド**

```bash
conda activate graphrag
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**ターミナル2: フロントエンド**

```bash
conda activate graphrag
cd frontend
npm start
```

## アクセス先

| URL | 用途 |
|---|---|
| `http://localhost:3000` | React UI |
| `http://localhost` | Apache経由 |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:7474` | Neo4j Browser |

## .env 設定リファレンス

```env
# LLMプロバイダ
PROVIDER=ollama

# ストレージ
STORAGE=neo4j

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EXTRACT_MODEL=qwen2.5:7b

# Gemini（https://aistudio.google.com/apikey で無料取得）
GEMINI_API_KEY=
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EXTRACT_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=models/gemini-embedding-001

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## LangChain 統合の設計思想

全コンポーネントが `config.py` 経由でLLM/Embeddingsを取得する。
プロバイダ固有のコードは `config.py` 内に閉じており、他のファイルは一切変更不要。

```
config.py
  ├─ get_chat_llm()     → graph_agent.py（回答生成・品質チェック）
  ├─ get_extract_llm()  → entity_extractor.py（エンティティ抽出）
  └─ get_embeddings()   → graph_rag.py / neo4j_store.py（ベクトル検索）
```

## Neo4j データモデル

```
(:Entity {name, type, embedding})
  -[:RELATES {relation}]->
(:Entity)

(:Chunk {chunk_id, text, doc_id, seq, embedding})
  -[:NEXT_CHUNK]->
(:Chunk)
  -[:SHARED_ENTITY]->
(:Chunk)
```

### ベクトルインデックス

- `chunk_embedding_index` → `:Chunk.embedding` (cosine, 768dim)
- `entity_embedding_index` → `:Entity.embedding` (cosine, 768dim)

### Cypher クエリ例

```cypher
-- 全エンティティの関係を可視化
MATCH (a:Entity)-[r:RELATES]->(b:Entity)
RETURN a, r, b

-- GNNから2ホップ以内のtechnologyエンティティ
MATCH (e:Entity {name: "GNN"})-[:RELATES*1..2]-(n:Entity)
WHERE n.type = "technology"
RETURN n.name, n.type
```

## API リファレンス

| メソッド | エンドポイント | 説明 |
|---|---|---|
| POST | `/api/chat` | チャット（非ストリーミング） |
| POST | `/api/chat/stream` | チャット（SSE） |
| POST | `/api/documents` | ドキュメント追加 |
| POST | `/api/entities/search` | エンティティベクトル検索 |
| GET | `/api/graph/stats` | グラフ統計 |
| GET | `/api/graph/data?query=` | グラフ可視化データ |
| POST | `/api/graph/reextract` | エンティティ再抽出 |
| POST | `/api/graph/reset` | エンティティグラフリセット |
| GET | `/api/health` | ヘルスチェック |

## トラブルシューティング

**hnswlib インストールエラー** → `pip install chroma-hnswlib` でプリビルド版を使用。

**uvicorn が見つからない** → hnswlibエラーで pip install が中断している。先に解決してから再実行。

**npm が見つからない** → `conda install nodejs -y` でインストール。

**GraphView.js 大文字小文字エラー** → `ren GraphView.js _temp.js` → `ren _temp.js GraphView.js`

**eslint exhaustive-deps エラー** → GraphView.js から該当コメント行を手動削除。

**グラフデータ取得失敗（ファイルモード）** → `backend/data/` を削除してuvicorn再起動。

**グラフデータ取得失敗（Neo4jモード）** → Neo4j起動確認。`http://localhost:7474` でアクセス確認。

**Neo4j ベクトルインデックスエラー** → Neo4j 5.11以上が必要。
