"""
GraphRAG 検索エンジン
- NetworkX: 知識グラフ管理 + コミュニティ検出
- hnswlib: 高速ANN検索
- Ollama nomic-embed-text: 埋め込みベクトル生成
"""

import networkx as nx
import hnswlib
import numpy as np
import httpx
import json
import hashlib
from pathlib import Path
from typing import Optional

# --- 設定 ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
DATA_DIR = Path("./data")
GRAPH_PATH = DATA_DIR / "knowledge_graph.graphml"
ENTITY_GRAPH_PATH = DATA_DIR / "entity_graph.graphml"
INDEX_PATH = DATA_DIR / "hnsw_index.bin"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# シングルトン
_retriever_instance: Optional["GraphRAGRetriever"] = None


def get_retriever() -> "GraphRAGRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GraphRAGRetriever()
        _retriever_instance.load_or_init()
    return _retriever_instance


class GraphRAGRetriever:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_graph = nx.DiGraph()  # エンティティ関係グラフ（有向）
        self.index: Optional[hnswlib.Index] = None
        self.chunks: list[dict] = []  # {id, text, metadata, node_id}
        self.communities: dict[str, list[str]] = {}  # community_id -> [node_ids]
        self._chunk_id_counter = 0

    # ========================================
    # 初期化・永続化
    # ========================================
    def load_or_init(self):
        """保存済みデータがあれば読込、なければ初期化"""
        DATA_DIR.mkdir(exist_ok=True)

        if GRAPH_PATH.exists() and INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.graph = nx.read_graphml(str(GRAPH_PATH))
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.index = hnswlib.Index(space="cosine", dim=EMBED_DIM)
            self.index.load_index(str(INDEX_PATH))
            self._chunk_id_counter = len(self.chunks)
            self._detect_communities()
            print(f"[GraphRAG] Loaded: {len(self.chunks)} chunks, "
                  f"{self.graph.number_of_nodes()} nodes, "
                  f"{len(self.communities)} communities")
        else:
            self.index = hnswlib.Index(space="cosine", dim=EMBED_DIM)
            self.index.init_index(max_elements=10000, ef_construction=200, M=16)
            self.index.set_ef(50)
            print("[GraphRAG] Initialized empty index")

        # エンティティグラフ読込
        if ENTITY_GRAPH_PATH.exists():
            self.entity_graph = nx.read_graphml(str(ENTITY_GRAPH_PATH))
            # DiGraphとして読み直す
            if not isinstance(self.entity_graph, nx.DiGraph):
                self.entity_graph = nx.DiGraph(self.entity_graph)
            print(f"[GraphRAG] Entity graph: {self.entity_graph.number_of_nodes()} entities, "
                  f"{self.entity_graph.number_of_edges()} relations")

    def save(self):
        """永続化"""
        DATA_DIR.mkdir(exist_ok=True)
        nx.write_graphml(self.graph, str(GRAPH_PATH))
        nx.write_graphml(self.entity_graph, str(ENTITY_GRAPH_PATH))
        self.index.save_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    # ========================================
    # ドキュメント追加
    # ========================================
    def add_document(self, text: str, metadata: dict = {}) -> str:
        """テキストをチャンク分割 → 埋め込み → グラフ追加"""
        chunks = self._split_text(text)
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]

        for i, chunk_text in enumerate(chunks):
            chunk_id = self._chunk_id_counter
            self._chunk_id_counter += 1

            # 埋め込み生成
            embedding = self._embed(chunk_text)

            # hnswlibに追加（インデックス拡張が必要な場合）
            current_max = self.index.get_max_elements()
            if chunk_id >= current_max:
                self.index.resize_index(current_max + 5000)
            self.index.add_items(
                np.array([embedding], dtype=np.float32),
                np.array([chunk_id]),
            )

            # チャンクを保存
            node_id = f"chunk_{doc_id}_{i}"
            self.chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata,
                "node_id": node_id,
                "doc_id": doc_id,
            })

            # グラフにノード追加
            self.graph.add_node(node_id, text=chunk_text[:100], doc_id=doc_id)

            # 同一ドキュメント内の隣接チャンクをエッジで接続
            if i > 0:
                prev_node = f"chunk_{doc_id}_{i-1}"
                self.graph.add_edge(prev_node, node_id, relation="next_chunk")

        # エンティティ間のエッジ構築（簡易版: 共通キーワード）
        self._build_entity_edges(doc_id)

        # LLMでエンティティ抽出 → エンティティグラフに追加
        self._extract_and_add_entities(text)

        # コミュニティ再検出
        self._detect_communities()

        # 保存
        self.save()

        return doc_id

    # ========================================
    # 検索
    # ========================================
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """ベクトル検索 + グラフ近傍展開"""
        if len(self.chunks) == 0:
            return []

        # 1. ベクトル検索
        query_vec = self._embed(query)
        k = min(top_k, len(self.chunks))
        labels, distances = self.index.knn_query(
            np.array([query_vec], dtype=np.float32), k=k
        )

        results = []
        seen_nodes = set()

        for label, dist in zip(labels[0], distances[0]):
            idx = int(label)
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            node_id = chunk["node_id"]
            seen_nodes.add(node_id)

            # コミュニティ特定
            community = self._get_community(node_id)

            results.append({
                "text": chunk["text"],
                "score": float(1 - dist),  # cosine similarity
                "node_id": node_id,
                "community": community,
                "source": "vector",
            })

        # 2. グラフ近傍展開（1ホップ隣接ノード）
        graph_neighbors = []
        for node_id in list(seen_nodes):
            if node_id in self.graph:
                for neighbor in self.graph.neighbors(node_id):
                    if neighbor not in seen_nodes:
                        # 隣接ノードに対応するチャンクを検索
                        for chunk in self.chunks:
                            if chunk["node_id"] == neighbor:
                                community = self._get_community(neighbor)
                                graph_neighbors.append({
                                    "text": chunk["text"],
                                    "score": 0.5,  # グラフ近傍は固定スコア
                                    "node_id": neighbor,
                                    "community": community,
                                    "source": "graph_neighbor",
                                })
                                seen_nodes.add(neighbor)
                                break

        # ベクトル結果 + グラフ近傍を統合（スコア順）
        results.extend(graph_neighbors[:3])
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def get_community_summary(self, community_id: str) -> str:
        """コミュニティのテキスト要約を生成"""
        if community_id not in self.communities:
            return ""

        node_ids = self.communities[community_id]
        texts = []
        for nid in node_ids[:5]:
            for chunk in self.chunks:
                if chunk["node_id"] == nid:
                    texts.append(chunk["text"][:200])
                    break

        if not texts:
            return ""

        return f"コミュニティ '{community_id}' ({len(node_ids)}ノード): " + " | ".join(texts)

    # ========================================
    # エンティティ抽出 → エンティティグラフ構築
    # ========================================
    def _extract_and_add_entities(self, text: str):
        """LLMでエンティティとリレーションを抽出し、entity_graphに追加"""
        from entity_extractor import extract_entities

        result = extract_entities(text)
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        if not entities:
            print("[GraphRAG] No entities extracted")
            return

        # エンティティをノードとして追加（重複時はスキップ）
        for ent in entities:
            name = ent["name"]
            etype = ent["type"]
            if name in self.entity_graph.nodes:
                # 既存ノードのタイプが未設定なら更新
                if not self.entity_graph.nodes[name].get("type"):
                    self.entity_graph.nodes[name]["type"] = etype
            else:
                self.entity_graph.add_node(name, type=etype)

        # リレーションをエッジとして追加
        entity_names = {e["name"] for e in entities}
        for rel in relations:
            src = rel["source"]
            tgt = rel["target"]
            label = rel["relation"]
            # ソースとターゲットがエンティティに存在する場合のみ
            if src in entity_names and tgt in entity_names:
                self.entity_graph.add_edge(src, tgt, relation=label)

        print(f"[GraphRAG] Extracted {len(entities)} entities, {len(relations)} relations")

    # ========================================
    # 内部ユーティリティ
    # ========================================
    def _embed(self, text: str) -> np.ndarray:
        """Ollama nomic-embed-text で埋め込み生成"""
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            vec = resp.json()["embedding"]

        arr = np.array(vec, dtype=np.float32)

        # 次元チェック（nomic-embed-text = 768次元）
        if len(arr) != EMBED_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(arr)}, expected {EMBED_DIM}. "
                f"Check EMBED_MODEL ({EMBED_MODEL}) settings."
            )
        return arr

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """テキストをチャンク分割（句点・改行を優先的に切る）"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # 句点（。）や改行の位置で切る
                for sep in ["。\n", "。", "\n\n", "\n", "、"]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks

    def _build_entity_edges(self, doc_id: str):
        """同一ドキュメント内のチャンク間でキーワードベースのエッジを構築"""
        doc_chunks = [c for c in self.chunks if c.get("doc_id") == doc_id]

        for i, c1 in enumerate(doc_chunks):
            for c2 in doc_chunks[i + 2:]:  # 隣接以外のチャンク
                # 共通する漢字2文字以上の語をカウント（簡易版）
                words1 = set(self._extract_keywords(c1["text"]))
                words2 = set(self._extract_keywords(c2["text"]))
                common = words1 & words2
                if len(common) >= 2:
                    self.graph.add_edge(
                        c1["node_id"],
                        c2["node_id"],
                        relation="shared_entity",
                        weight=len(common),
                    )

    def _extract_keywords(self, text: str) -> list[str]:
        """簡易キーワード抽出（漢字の連続を取り出す）"""
        import re
        return re.findall(r"[\u4e00-\u9fff]{2,}", text)

    def _detect_communities(self):
        """Louvain法でコミュニティ検出"""
        if self.graph.number_of_nodes() < 2:
            self.communities = {}
            return

        try:
            from networkx.algorithms.community import louvain_communities
            comms = louvain_communities(self.graph, seed=42)
            self.communities = {}
            for i, comm in enumerate(comms):
                comm_id = f"community_{i}"
                self.communities[comm_id] = list(comm)
                for node in comm:
                    self.graph.nodes[node]["community"] = comm_id
        except Exception:
            self.communities = {"all": list(self.graph.nodes())}

    def _get_community(self, node_id: str) -> str:
        """ノードが属するコミュニティIDを取得"""
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id].get("community", "unknown")
        return "unknown"