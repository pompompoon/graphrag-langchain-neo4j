# GraphRAG Local LLM

ローカル環境で動作する、知識グラフベースのRAG（Retrieval-Augmented Generation）チャットアプリケーション。
テキストからエンティティ（人物・組織・技術・概念など）を自動抽出し、知識グラフとして構築・可視化する。

## アーキテクチャ

```
React (localhost:3000)
  ↓
Apache (localhost:80) ← リバースプロキシ
  ↓
FastAPI (localhost:8000)
  ↓
LangGraph Agent
  ├─ クエリ分析ノード ......... LLMで検索クエリを最適化
  ├─ GraphRAG検索ノード ....... NetworkX + hnswlib でベクトル＋グラフ検索
  ├─ Ollama回答生成ノード ..... qwen2.5:7b で回答生成
  └─ 品質チェックノード ....... LLM自己評価 → 品質NGならループ再検索
```

## 主な機能

- **エンティティ抽出**: Ollama LLMがテキストから固有名詞・専門用語を単語単位で抽出
- **知識グラフ構築**: 抽出したエンティティとリレーションをNetworkXの有向グラフとして管理
- **グラフ可視化**: ダークテーマのCanvas力学シミュレーションでインタラクティブに表示
  - ノードドラッグ / ズーム / パン
  - エンティティタイプ別アイコン・色分け（👤 person, 🏛 organization, ⚡ technology 等）
  - エッジにリレーションラベル表示
  - グラフ統計パネル（密度・連結成分・中心性ランキング）
- **RAGチャット**: LangGraphエージェントによる品質チェックループ付きの自律的な検索→回答生成
- **SSEストリーミング**: 各ノードの処理状況をリアルタイム表示

## 技術スタック

| レイヤー | 技術 |
|---|---|
| フロントエンド | React 18, Canvas (力学シミュレーション) |
| リバースプロキシ | Apache (mod_proxy) |
| APIサーバー | FastAPI, Uvicorn |
| エージェント | LangGraph (LangChain) |
| 知識グラフ | NetworkX (有向グラフ) |
| ベクトル検索 | hnswlib (ANN), nomic-embed-text (768次元) |
| LLM | Ollama (qwen2.5:7b) |
| エンティティ抽出 | Ollama LLM + Few-shotプロンプト |

## ディレクトリ構成

```
graphrag-app/
├── backend/
│   ├── main.py                 # FastAPI エンドポイント定義
│   ├── graph_agent.py          # LangGraph エージェント（4ノード構成）
│   ├── graph_rag.py            # GraphRAG検索エンジン（NetworkX + hnswlib）
│   ├── entity_extractor.py     # LLMエンティティ抽出器
│   ├── requirements.txt        # Python依存パッケージ
│   └── data/                   # 永続化データ（自動生成）
│       ├── knowledge_graph.graphml
│       ├── entity_graph.graphml
│       ├── hnsw_index.bin
│       └── chunks.json
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js
│       ├── App.js              # メインUI（チャット + サイドバー）
│       └── GraphView.js        # グラフ可視化（Canvas + 力学モデル）
├── config/
│   └── httpd-graphrag.conf     # Apache リバースプロキシ設定
├── start.bat                   # Windows 一括起動スクリプト
└── README.md
```

## セットアップ

### 前提条件

- Windows 10/11
- Anaconda (Python 3.11)
- Ollama
- Apache (XAMPP等)
- Node.js (conda経由でもOK)

### 1. Ollama モデル準備

```bash
ollama pull qwen2.5:7b

ollama pull nomic-embed-text
set OLLAMA_MODEL=
```

### 2. Python環境構築

```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
cd backend

# hnswlib はプリビルド版を使用（C++ Build Tools不要）
pip install chroma-hnswlib

# 残りのパッケージ
pip install fastapi uvicorn[standard] pydantic langchain langchain-ollama langgraph networkx numpy python-dotenv httpx
```

### 3. Node.js 準備

```bash
conda install nodejs -y
cd frontend
npm install
```

### 4. Apache 設定

`config/httpd-graphrag.conf` を Apache の `conf/extra/` にコピーし、
`httpd.conf` に以下を追加:

```apache
Include conf/extra/httpd-graphrag.conf
```

また、以下のモジュールが有効であることを確認:

```apache
LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so
LoadModule rewrite_module modules/mod_rewrite.so
LoadModule headers_module modules/mod_headers.so
```

## 起動方法

### 一括起動（Windows）

```bash
start.bat
```

### 個別起動

ターミナルを3つ開き、それぞれ実行:

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

**ターミナル3: Apache**
```bash
httpd -k restart
```

## アクセス先

| URL | 用途 |
|---|---|
| http://localhost:3000 | React UI（直接） |
| http://localhost | Apache経由（本番想定） |
| http://localhost:8000/docs | FastAPI Swagger UI |

## 使い方

### ドキュメント追加

サイドバーのテキストエリアにテキストを貼り付けて「グラフに追加」ボタンを押す。
自動的にエンティティ抽出 → グラフ構築 → ベクトルインデックス追加が行われる。

### グラフ可視化

ヘッダーの「グラフ」タブに切り替えると、抽出されたエンティティとリレーションが
力学シミュレーションで表示される。

### チャット

ヘッダーの「チャット」タブで質問を入力。LangGraphエージェントが以下のフローで回答:

1. クエリ分析 — LLMが検索キーワードを最適化
2. GraphRAG検索 — ベクトル検索 + グラフ近傍展開
3. 回答生成 — 検索結果をコンテキストとしてLLMが回答
4. 品質チェック — LLMが自己評価（スコア 0.6 未満なら再検索ループ、最大2回）

## API リファレンス

| メソッド | エンドポイント | 説明 |
|---|---|---|
| POST | `/api/chat` | チャット（非ストリーミング） |
| POST | `/api/chat/stream` | チャット（SSEストリーミング） |
| POST | `/api/documents` | ドキュメント追加 |
| GET | `/api/graph/stats` | グラフ統計情報 |
| GET | `/api/graph/data` | グラフ可視化データ（ノード+エッジ+統計） |
| POST | `/api/graph/reextract` | 既存ドキュメントからエンティティ再抽出 |
| POST | `/api/graph/reset` | エンティティグラフをリセット |
| GET | `/api/health` | ヘルスチェック（Ollama接続確認） |

## トラブルシューティング

### hnswlib インストールエラー

```
error: Microsoft Visual C++ 14.0 or greater is required.
```

→ `pip install chroma-hnswlib` でプリビルド版を使用する。

### uvicorn が見つからない

→ `pip install` が途中で失敗している。hnswlib問題を先に解決してから再実行。

### グラフデータの取得に失敗

→ `backend/data/` フォルダの中身を削除して uvicorn を再起動:
```bash
Remove-Item backend\data\* -Force
```

### npm が見つからない

→ `conda install nodejs -y` で Node.js をインストール。

### GraphView.js の大文字小文字エラー

→ Windows でファイル名を2段階でリネーム:
```bash
ren GraphView.js _temp.js
ren _temp.js GraphView.js
```