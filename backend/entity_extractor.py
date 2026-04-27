"""
エンティティ抽出器（LangChain版）
httpx直接呼び出しを排除 → config.py経由でOllama/Geminiを自動切り替え
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_extract_llm

EXTRACT_PROMPT = """あなたは知識グラフ構築のためのエンティティ抽出エキスパートです。
テキストから**個別の単語・用語**をエンティティとして抽出し、それらの関係を定義してください。

## 重要なルール
- エンティティは**1つの固有名詞や専門用語**にする（文章やフレーズにしないこと）
- 例: ✅ "GNN" ✅ "NetworkX" ✅ "視野検査" ✅ "Anthropic"
- 例: ❌ "GNNを用いた分析手法" ❌ "視野検査のTOP戦略について"
- できるだけ多くのエンティティを抽出する（最低10個以上を目指す）
- リレーションは2〜4文字の短い動詞で表現する

## エンティティの type
- person: 人名
- organization: 組織・企業
- technology: 技術・手法・ツール・アルゴリズム
- location: 場所
- event: イベント・会議
- concept: 概念・分野・指標
- date: 日時・期間

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
{text}"""


def _build_chain():
    llm = get_extract_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT)
    return prompt | llm | StrOutputParser()


def extract_entities(text: str) -> dict:
    chunks = _split_for_extraction(text, max_len=1500)
    all_entities = {}
    all_relations = []
    seen_relations = set()
    chain = _build_chain()

    for chunk in chunks:
        result = _extract_from_chunk(chain, chunk)
        for ent in result.get("entities", []):
            name = ent["name"].strip()
            if name and len(name) <= 50:
                if name not in all_entities:
                    all_entities[name] = ent
        for rel in result.get("relations", []):
            key = (rel["source"], rel["target"], rel["relation"])
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(rel)

    entity_names = set(all_entities.keys())
    valid_relations = [
        r for r in all_relations
        if r["source"] in entity_names and r["target"] in entity_names
    ]
    print(f"[EntityExtractor] {len(chunks)} chunks -> "
          f"{len(all_entities)} entities, {len(valid_relations)} relations")
    return {"entities": list(all_entities.values()), "relations": valid_relations}


def _split_for_extraction(text, max_len=1500):
    if len(text) <= max_len:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_len
        if end < len(text):
            for sep in ["。\n", "。", "\n\n", "\n"]:
                pos = text[start:end].rfind(sep)
                if pos > max_len // 2:
                    end = start + pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _extract_from_chunk(chain, text):
    try:
        raw = chain.invoke({"text": text})
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
        return {
            "entities": [e for e in data.get("entities", [])
                         if isinstance(e, dict) and "name" in e and e.get("type") in valid_types],
            "relations": [r for r in data.get("relations", [])
                          if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r],
        }
    except Exception as e:
        print(f"[EntityExtractor] Chunk error: {e}")
        return {"entities": [], "relations": []}
