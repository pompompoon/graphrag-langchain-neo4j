# GraphRAG Local LLM チューニングガイド

精度向上のためのパラメータとアプローチを4領域に分けて解説する。

---

## 1. エンティティ抽出の精度

**ファイル: `entity_extractor.py`**

### 1-1. LLMモデルの変更（効果: ★★★）

```python
# 現在
OLLAMA_MODEL = "qwen2.5:7b"

# 精度重視（VRAM 16GB以上）
OLLAMA_MODEL = "qwen2.5:14b"

# 日本語特化
OLLAMA_MODEL = "elyza:13b"

# コーディング・技術文書に強い
OLLAMA_MODEL = "codestral:22b"
```

モデルが大きいほど抽出精度は上がるが、速度は下がる。
ドキュメント追加時のみ使うので、多少遅くても大きいモデルを推奨。

### 1-2. プロンプトチューニング（効果: ★★★）

```python
# entity_extractor.py の EXTRACT_PROMPT を改善

# ① ドメイン特化の例を追加（医療AI向けの例）
## 抽出例
入力: 「HFAの感度値をGATv2Convで予測し、視野の重要度マップを生成する」
出力:
{{"entities": [
  {{"name": "HFA", "type": "technology"}},
  {{"name": "感度値", "type": "concept"}},
  {{"name": "GATv2Conv", "type": "technology"}},
  {{"name": "視野", "type": "concept"}},
  {{"name": "重要度マップ", "type": "concept"}}
],
"relations": [
  {{"source": "GATv2Conv", "target": "感度値", "relation": "予測する"}},
  {{"source": "GATv2Conv", "target": "重要度マップ", "relation": "生成する"}},
  {{"source": "HFA", "target": "感度値", "relation": "測定する"}}
]}}

# ② ネガティブ例を増やす
- ❌ 「を用いて」「による」「に関する」を含む長いフレーズは不可
- ❌ 助詞や動詞を含むものはエンティティではない

# ③ temperature を下げる（現在 0.1 → 0.0 に）
"options": {"temperature": 0.0}
```

### 1-3. チャンク分割サイズ（効果: ★★）

```python
# entity_extractor.py の _split_for_extraction
# 小さいチャンク = 1チャンクあたりの抽出精度↑、全体の処理時間↑

# 現在
max_len=1500

# 精度重視
max_len=800   # より細かく分割→抽出漏れが減る

# 速度重視
max_len=2000  # 大きく分割→処理回数が減る
```

### 1-4. 後処理フィルタ（効果: ★★）

```python
# entity_extractor.py の extract_entities に追加

# 不要なエンティティを除外するフィルタ
STOP_WORDS = {"こと", "もの", "ため", "場合", "方法", "結果", "以下", "上記"}

entities = [
    e for e in entities
    if e["name"] not in STOP_WORDS
    and len(e["name"]) >= 2          # 1文字は除外
    and len(e["name"]) <= 30         # 長すぎるものも除外
    and not e["name"].endswith("する")  # 動詞は除外
    and not e["name"].endswith("的")   # 形容詞も除外
]
```

### 1-5. エンティティの正規化（効果: ★★）

```python
# 同義語の統合（entity_extractor.py に追加）
NORMALIZE_MAP = {
    "GATv2": "GATv2Conv",
    "Graph Attention Network": "GAT",
    "NN": "ニューラルネットワーク",
}

def normalize_entity(name):
    return NORMALIZE_MAP.get(name, name)
```

---

## 2. 検索（ベクトル+グラフ）の精度

**ファイル: `graph_rag.py`**

### 2-1. 埋め込みモデルの変更（効果: ★★★）

```python
# 現在
EMBED_MODEL = "nomic-embed-text"   # 768次元、英語寄り
EMBED_DIM = 768

# 日本語に強い代替（ollama pull してから変更）
EMBED_MODEL = "bge-m3"             # 1024次元、多言語対応
EMBED_DIM = 1024

EMBED_MODEL = "multilingual-e5-large"  # 1024次元
EMBED_DIM = 1024
```

**注意:** モデルを変更したら `backend/data/` を全削除して再構築が必要。

### 2-2. hnswlibパラメータ（効果: ★★）

```python
# graph_rag.py の load_or_init

# 現在
self.index.init_index(max_elements=10000, ef_construction=200, M=16)
self.index.set_ef(50)

# 精度重視（構築は遅くなるが検索精度↑）
self.index.init_index(max_elements=10000, ef_construction=400, M=32)
self.index.set_ef(100)

# ef_construction: インデックス構築時の探索幅（大→精度↑、構築速度↓）
# M: 各ノードの最大接続数（大→精度↑、メモリ↑）
# ef: 検索時の探索幅（大→精度↑、検索速度↓）
```

### 2-3. チャンク分割の最適化（効果: ★★★）

```python
# graph_rag.py の _split_text

# 現在
chunk_size=500, overlap=50

# 精度重視（小さいチャンク + 大きいオーバーラップ）
chunk_size=300, overlap=100

# 理由:
# - 小さいチャンク → 検索結果がより的確（不要な文が混ざりにくい）
# - 大きいオーバーラップ → 文脈の切れ目を防ぐ
# - ただしチャンク数が増えるのでインデックスサイズ↑
```

### 2-4. グラフ近傍展開の深さ（効果: ★★）

```python
# graph_rag.py の search メソッド

# 現在: 1ホップのみ
for neighbor in self.graph.neighbors(node_id):

# 2ホップまで展開（関連性が広がるが、ノイズも増える）
for n1 in self.graph.neighbors(node_id):
    if n1 not in seen_nodes:
        # 1ホップ目を追加（スコア0.4）
        ...
        for n2 in self.graph.neighbors(n1):
            if n2 not in seen_nodes:
                # 2ホップ目を追加（スコア0.2）
                ...
```

### 2-5. リランキング（効果: ★★★）

```python
# graph_rag.py の search メソッドの最後に追加

def _rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """LLMで検索結果をリランキング"""
    if len(results) <= top_k:
        return results

    candidates = "\n".join([
        f"[{i}] {r['text'][:200]}" for i, r in enumerate(results)
    ])

    prompt = (
        f"質問: {query}\n\n"
        f"以下の候補から、質問への回答に最も関連性の高いものを"
        f"番号で{top_k}個選んでください。番号のみをカンマ区切りで出力:\n\n"
        f"{candidates}"
    )

    # LLMでリランキング
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
        )
        raw = resp.json()["response"]

    # 番号を抽出
    import re
    indices = [int(x) for x in re.findall(r'\d+', raw) if int(x) < len(results)]
    reranked = [results[i] for i in indices[:top_k]]

    return reranked if reranked else results[:top_k]
```

### 2-6. エンティティ検索のスコア閾値（効果: ★）

```python
# main.py の graph_data エンドポイント

# 現在
if er["score"] > 0.3:  # 類似度0.3以上のみハイライト

# 厳しくする（ノイズを減らす）
if er["score"] > 0.5:

# 緩くする（ヒット数を増やす）
if er["score"] > 0.2:
```

---

## 3. 回答生成（LLM）の品質

**ファイル: `graph_agent.py`**

### 3-1. 生成モデルの変更（効果: ★★★）

```python
# 現在
OLLAMA_MODEL = "qwen2.5:7b"

# 精度重視
OLLAMA_MODEL = "qwen2.5:14b"    # VRAM 16GB
OLLAMA_MODEL = "qwen2.5:32b"    # VRAM 24GB+
OLLAMA_MODEL = "llama3.1:8b"    # Meta製、英語に強い
OLLAMA_MODEL = "gemma2:9b"      # Google製

# 使い分け戦略:
# エンティティ抽出 → 大きいモデル（1回だけなので遅くてOK）
# チャット回答 → 中程度モデル（レスポンス速度も重要）
# 品質チェック → 小さいモデルでもOK（数値1つ出すだけ）
```

### 3-2. 品質チェックの閾値（効果: ★★）

```python
# 現在
MAX_RETRIES = 2
QUALITY_THRESHOLD = 0.6

# 厳しくする（品質↑、レスポンス時間↑）
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.75

# 緩くする（速度重視）
MAX_RETRIES = 1
QUALITY_THRESHOLD = 0.5
```

### 3-3. プロンプトの改善（効果: ★★★）

```python
# graph_agent.py の generate_answer ノード

# 改善版プロンプト
prompt = ChatPromptTemplate.from_template(
    "あなたは知識グラフを活用するAIアシスタントです。\n\n"
    "## 回答ルール\n"
    "1. 検索結果に含まれる情報のみを使って回答する\n"
    "2. 検索結果にない情報を推測で補わない\n"
    "3. 複数のソースが矛盾する場合はスコアの高い方を優先する\n"
    "4. 情報が不足している場合は「この情報は知識グラフに含まれていません」と明示\n"
    "5. 回答には根拠となったソースを引用する\n\n"
    "## 検索結果（スコア順）\n{context}\n\n"
    "## 質問\n{question}\n\n"
    "回答:"
)
```

### 3-4. コンテキストウィンドウの管理（効果: ★★）

```python
# graph_agent.py の generate_answer

# 検索結果が多すぎるとコンテキストが溢れる
# → スコア上位N件に絞る + テキストを切り詰め
context = "\n".join([
    f"[スコア:{r.get('score', 0):.2f}] {r['text'][:300]}"  # 300文字に制限
    for r in state["search_results"][:5]  # 上位5件のみ
])

# 会話履歴も直近4件に制限
history_text = "\n".join([
    f"{m['role']}: {m['content'][:200]}"
    for m in state["chat_history"][-4:]
])
```

---

## 4. システム全体のチューニング

### 4-1. 評価データセットの作成（効果: ★★★）

```python
# eval_dataset.py（新規作成）

EVAL_QA = [
    {
        "question": "GATv2Convとは何ですか？",
        "expected_entities": ["GATv2Conv", "GAT", "GNN"],
        "expected_answer_keywords": ["グラフ", "注意機構", "ノード"],
    },
    {
        "question": "視野検査のTOP戦略の特徴は？",
        "expected_entities": ["TOP戦略", "視野検査", "HFA"],
        "expected_answer_keywords": ["閾値", "効率", "測定"],
    },
]

def evaluate_retrieval(retriever, qa_pairs):
    """検索精度を定量評価"""
    results = []
    for qa in qa_pairs:
        search_results = retriever.search(qa["question"])
        found_entities = set()
        for r in search_results:
            for ent in qa["expected_entities"]:
                if ent in r["text"]:
                    found_entities.add(ent)

        recall = len(found_entities) / len(qa["expected_entities"])
        results.append({"question": qa["question"], "recall": recall})

    avg_recall = sum(r["recall"] for r in results) / len(results)
    print(f"平均Recall: {avg_recall:.3f}")
    return results
```

### 4-2. モデル別の使い分け（効果: ★★★）

```python
# 各コンポーネントで異なるモデルを使用

# entity_extractor.py → 大きいモデル（精度重視）
OLLAMA_MODEL = "qwen2.5:14b"

# graph_agent.py → 中程度モデル（バランス）
OLLAMA_MODEL = "qwen2.5:7b"

# 品質チェック → 小さいモデルでOK
# graph_agent.py の check_quality 内で別モデル指定
llm_checker = ChatOllama(model="qwen2.5:3b", ...)
```

### 4-3. パラメータ一覧表（現在値 → 推奨値）

| パラメータ | ファイル | 現在値 | 精度重視 | 速度重視 |
|---|---|---|---|---|
| 生成モデル | graph_agent.py | qwen2.5:7b | qwen2.5:14b | qwen2.5:3b |
| 抽出モデル | entity_extractor.py | qwen2.5:7b | qwen2.5:14b | qwen2.5:7b |
| 埋め込みモデル | graph_rag.py | nomic-embed-text | bge-m3 | nomic-embed-text |
| チャンクサイズ | graph_rag.py | 500 | 300 | 800 |
| オーバーラップ | graph_rag.py | 50 | 100 | 30 |
| hnswlib ef | graph_rag.py | 50 | 100 | 30 |
| hnswlib M | graph_rag.py | 16 | 32 | 12 |
| 品質閾値 | graph_agent.py | 0.6 | 0.75 | 0.4 |
| リトライ上限 | graph_agent.py | 2 | 3 | 1 |
| 検索top_k | graph_rag.py | 5 | 8 | 3 |
| 抽出チャンク | entity_extractor.py | 1500 | 800 | 2000 |
| temperature | entity_extractor.py | 0.1 | 0.0 | 0.1 |

### 4-4. 即効性のある3ステップ

1. **埋め込みモデル変更** → `bge-m3` に変えるだけで日本語の検索精度が大幅向上
2. **チャンクサイズ縮小** → 500→300 で検索のピンポイント精度↑
3. **リランキング追加** → LLMで検索結果を再評価、ノイズ除去