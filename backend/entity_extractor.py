"""
エンティティ抽出器
Ollama LLM でテキストからエンティティ（人物・組織・技術・場所・イベント等）と
リレーション（関係）を抽出し、知識グラフに追加する
"""

import httpx
import json
import re

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

EXTRACT_PROMPT = """あなたは知識グラフ構築のためのエンティティ抽出エキスパートです。
テキストから**個別の単語・用語**をエンティティとして抽出し、それらの関係を定義してください。

## 重要なルール
- エンティティは**1つの固有名詞や専門用語**にする（文章やフレーズにしないこと）
- 例: ✅ "GNN" ✅ "NetworkX" ✅ "視野検査" ✅ "Anthropic"
- 例: ❌ "GNNを用いた分析手法" ❌ "視野検査のTOP戦略について"
- できるだけ多くのエンティティを抽出する（最低10個以上を目指す）
- 略称と正式名称は別エンティティとして抽出してよい
- リレーションは2〜4文字の短い動詞で表現する

## エンティティの type
- person: 人名（例: "岩本武", "Hinton"）
- organization: 組織・企業（例: "Google", "Findex", "東京大学"）
- technology: 技術・手法・ツール・アルゴリズム（例: "GNN", "PyTorch", "hnswlib", "Louvain法"）
- location: 場所（例: "東京", "ap-northeast-1"）
- event: イベント・会議（例: "NeurIPS", "ICML2024"）
- concept: 概念・分野・指標（例: "グラフマイニング", "中心性", "精度"）
- date: 日時・期間（例: "2024年", "Q3"）

## 抽出例
入力: 「GNNはPyTorch Geometricで実装され、NetworkXのグラフ構造をもとにノード分類を行う」
出力:
{{"entities": [
  {{"name": "GNN", "type": "technology"}},
  {{"name": "PyTorch Geometric", "type": "technology"}},
  {{"name": "NetworkX", "type": "technology"}},
  {{"name": "ノード分類", "type": "concept"}}
],
"relations": [
  {{"source": "GNN", "target": "PyTorch Geometric", "relation": "実装に使用"}},
  {{"source": "GNN", "target": "NetworkX", "relation": "入力として使用"}},
  {{"source": "GNN", "target": "ノード分類", "relation": "実行する"}}
]}}

## 出力形式（JSONのみ出力、他の文字は絶対に出力しないこと）
{{"entities": [...], "relations": [...]}}

## テキスト
{text}

## JSON出力:"""


def extract_entities(text: str) -> dict:
    """テキストからエンティティとリレーションを抽出（長文は分割処理）"""
    # 長文は分割して各チャンクから抽出 → マージ
    chunks = _split_for_extraction(text, max_len=1500)
    all_entities = {}  # name -> entity dict（重複排除）
    all_relations = []
    seen_relations = set()

    for chunk in chunks:
        result = _extract_from_chunk(chunk)
        for ent in result.get("entities", []):
            name = ent["name"].strip()
            if name and len(name) <= 50:  # 長すぎるものは除外
                if name not in all_entities:
                    all_entities[name] = ent
        for rel in result.get("relations", []):
            key = (rel["source"], rel["target"], rel["relation"])
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(rel)

    # リレーションのsource/targetがエンティティに存在するか検証
    entity_names = set(all_entities.keys())
    valid_relations = [
        r for r in all_relations
        if r["source"] in entity_names and r["target"] in entity_names
    ]

    print(f"[EntityExtractor] {len(chunks)} chunks → "
          f"{len(all_entities)} entities, {len(valid_relations)} relations")

    return {
        "entities": list(all_entities.values()),
        "relations": valid_relations,
    }


def _split_for_extraction(text: str, max_len: int = 1500) -> list[str]:
    """抽出用にテキストを分割"""
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        if end < len(text):
            # 句点で切る
            for sep in ["。\n", "。", "\n\n", "\n"]:
                pos = text[start:end].rfind(sep)
                if pos > max_len // 2:
                    end = start + pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _extract_from_chunk(text: str) -> dict:
    """1チャンクからエンティティ抽出"""
    prompt = EXTRACT_PROMPT.format(text=text)

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            resp.raise_for_status()
            raw = resp.json()["response"]

        # JSONを抽出
        raw = raw.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            raw = raw[json_start:json_end]

        data = json.loads(raw)

        valid_types = {"person", "organization", "technology",
                       "location", "event", "concept", "date"}
        entities = [
            e for e in data.get("entities", [])
            if isinstance(e, dict) and "name" in e and e.get("type") in valid_types
        ]
        relations = [
            r for r in data.get("relations", [])
            if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r
        ]

        return {"entities": entities, "relations": relations}

    except (json.JSONDecodeError, KeyError, httpx.HTTPError) as e:
        print(f"[EntityExtractor] Chunk error: {e}")
        return {"entities": [], "relations": []}