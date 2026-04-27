"""
Neo4j ストレージバックエンド
GraphML + hnswlib + JSON → Neo4j に一元化

ノードラベル:
  :Chunk    - テキストチャンク（doc_id, text, embedding）
  :Entity   - 抽出エンティティ（name, type, embedding）

リレーション:
  (:Chunk)-[:NEXT_CHUNK]->(:Chunk)           - 隣接チャンク
  (:Chunk)-[:SHARED_ENTITY]->(:Chunk)        - 共通キーワード
  (:Entity)-[:RELATES {relation: "..."}]->(:Entity) - エンティティ間の関係
  (:Entity)-[:MENTIONED_IN]->(:Chunk)        - エンティティがチャンクに出現

ベクトルインデックス（Neo4j 5.11+）:
  chunk_embedding_index   - チャンクのベクトル検索
  entity_embedding_index  - エンティティのベクトル検索
"""

import hashlib
import numpy as np
from typing import Optional
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, get_embeddings, get_embed_dim


class Neo4jStore:
    def __init__(self):
        self._driver = None
        self._embeddings = None
        self._embed_dim = get_embed_dim()

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        return self._driver

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def close(self):
        if self._driver:
            self._driver.close()

    # ========================================
    # 初期化（インデックス・制約の作成）
    # ========================================
    def init_schema(self):
        """Neo4jスキーマとベクトルインデックスを作成"""
        with self.driver.session() as session:
            # ユニーク制約
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

            # ベクトルインデックス（Neo4j 5.11+）
            try:
                session.run("""
                    CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=self._embed_dim)

                session.run("""
                    CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=self._embed_dim)

                print("[Neo4j] Schema and vector indexes created")
            except Exception as e:
                print(f"[Neo4j] Vector index creation: {e}")
                print("[Neo4j] Note: Vector indexes require Neo4j 5.11+")

    # ========================================
    # ドキュメント追加
    # ========================================
    def add_document(self, text: str, metadata: dict = {}) -> str:
        """テキストをチャンク分割 → 埋め込み → Neo4jに格納"""
        chunks = self._split_text(text)
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]

        with self.driver.session() as session:
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"chunk_{doc_id}_{i}"
                embedding = self._embed(chunk_text)

                # Chunkノード作成
                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.doc_id = $doc_id,
                        c.seq = $seq,
                        c.source = $source,
                        c.embedding = $embedding
                """, chunk_id=chunk_id, text=chunk_text, doc_id=doc_id,
                     seq=i, source=metadata.get("source", ""),
                     embedding=embedding.tolist())

                # 隣接チャンクのエッジ
                if i > 0:
                    prev_id = f"chunk_{doc_id}_{i-1}"
                    session.run("""
                        MATCH (a:Chunk {chunk_id: $prev}), (b:Chunk {chunk_id: $curr})
                        MERGE (a)-[:NEXT_CHUNK]->(b)
                    """, prev=prev_id, curr=chunk_id)

            # 共通キーワードのエッジ
            self._build_shared_edges(session, doc_id)

        # エンティティ抽出
        self._extract_and_add_entities(text)

        print(f"[Neo4j] Document '{doc_id}': {len(chunks)} chunks indexed")
        return doc_id

    def _extract_and_add_entities(self, text: str):
        """LLMでエンティティ抽出 → Neo4jに格納"""
        from entity_extractor import extract_entities
        result = extract_entities(text)

        with self.driver.session() as session:
            for ent in result.get("entities", []):
                name = ent["name"]
                etype = ent.get("type", "concept")
                try:
                    embedding = self._embed(name)
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.embedding = $embedding
                    """, name=name, type=etype, embedding=embedding.tolist())
                except Exception as e:
                    print(f"[Neo4j] Entity embed error '{name}': {e}")

            for rel in result.get("relations", []):
                session.run("""
                    MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                    MERGE (a)-[r:RELATES]->(b)
                    SET r.relation = $rel
                """, src=rel["source"], tgt=rel["target"],
                     rel=rel.get("relation", ""))

    # ========================================
    # 検索
    # ========================================
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """統合検索: チャンクベクトル + エンティティベクトル + グラフ近傍"""
        query_vec = self._embed(query)
        results = []
        seen = set()

        with self.driver.session() as session:
            # 1. チャンクベクトル検索
            try:
                records = session.run("""
                    CALL db.index.vector.queryNodes(
                        'chunk_embedding_index', $k, $vec
                    ) YIELD node, score
                    RETURN node.chunk_id AS chunk_id,
                           node.text AS text,
                           node.doc_id AS doc_id,
                           score
                """, k=top_k, vec=query_vec.tolist()).data()

                for r in records:
                    results.append({
                        "text": r["text"], "score": r["score"],
                        "node_id": r["chunk_id"], "source": "vector",
                        "community": r["doc_id"],
                    })
                    seen.add(r["chunk_id"])
            except Exception as e:
                print(f"[Neo4j] Chunk vector search error: {e}")

            # 2. エンティティベクトル検索 → 関連チャンクをテキストマッチ
            entity_names = []
            try:
                ent_records = session.run("""
                    CALL db.index.vector.queryNodes(
                        'entity_embedding_index', $k, $vec
                    ) YIELD node, score
                    WHERE score > 0.3
                    RETURN node.name AS name, node.type AS type, score
                """, k=5, vec=query_vec.tolist()).data()

                entity_names = [r["name"] for r in ent_records]
            except Exception as e:
                print(f"[Neo4j] Entity vector search error: {e}")

            # エンティティ名を含むチャンクを追加
            for ename in entity_names:
                chunks = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.text CONTAINS $name AND NOT c.chunk_id IN $seen
                    RETURN c.chunk_id AS chunk_id, c.text AS text,
                           c.doc_id AS doc_id
                    LIMIT 3
                """, name=ename, seen=list(seen)).data()

                for c in chunks:
                    results.append({
                        "text": c["text"], "score": 0.6,
                        "node_id": c["chunk_id"], "source": "entity_match",
                        "community": c["doc_id"], "matched_entity": ename,
                    })
                    seen.add(c["chunk_id"])

            # 3. グラフ近傍展開（Cypherで1ホップ）
            if seen:
                neighbors = session.run("""
                    MATCH (a:Chunk)-[:NEXT_CHUNK|SHARED_ENTITY]-(b:Chunk)
                    WHERE a.chunk_id IN $ids AND NOT b.chunk_id IN $ids
                    RETURN DISTINCT b.chunk_id AS chunk_id,
                           b.text AS text, b.doc_id AS doc_id
                    LIMIT 3
                """, ids=list(seen)).data()

                for nb in neighbors:
                    results.append({
                        "text": nb["text"], "score": 0.4,
                        "node_id": nb["chunk_id"], "source": "graph_neighbor",
                        "community": nb["doc_id"],
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        for r in results:
            r["related_entities"] = entity_names[:5]
        return results[:top_k]

    def search_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """エンティティのベクトル検索 + グラフ近傍"""
        query_vec = self._embed(query)
        results = []

        with self.driver.session() as session:
            try:
                records = session.run("""
                    CALL db.index.vector.queryNodes(
                        'entity_embedding_index', $k, $vec
                    ) YIELD node, score
                    WITH node, score
                    OPTIONAL MATCH (node)-[r:RELATES]->(neighbor:Entity)
                    RETURN node.name AS name, node.type AS type, score,
                           collect({name: neighbor.name,
                                    relation: r.relation}) AS neighbors
                """, k=top_k, vec=query_vec.tolist()).data()

                for r in records:
                    neighbors = [n for n in r["neighbors"]
                                 if n.get("name") is not None]
                    results.append({
                        "name": r["name"], "type": r["type"],
                        "score": r["score"], "neighbors": neighbors,
                    })
            except Exception as e:
                print(f"[Neo4j] Entity search error: {e}")

        return results

    def get_community_summary(self, community_id: str) -> str:
        """doc_idベースのコミュニティ要約"""
        with self.driver.session() as session:
            records = session.run("""
                MATCH (c:Chunk {doc_id: $doc_id})
                RETURN c.text AS text
                ORDER BY c.seq LIMIT 5
            """, doc_id=community_id).data()
        texts = [r["text"][:200] for r in records]
        return f"ドキュメント '{community_id}': " + " | ".join(texts) if texts else ""

    # ========================================
    # グラフ可視化データ
    # ========================================
    def get_graph_data(self, query: str = "") -> dict:
        """エンティティグラフの可視化データ"""
        # 検索ハイライト
        highlighted = {}
        if query.strip():
            for er in self.search_entities(query, top_k=10):
                if er["score"] > 0.3:
                    highlighted[er["name"]] = er["score"]
                    for nb in er.get("neighbors", []):
                        if nb["name"] not in highlighted:
                            highlighted[nb["name"]] = er["score"] * 0.5

        type_colors = {
            "person": "#ef4444", "organization": "#2dd4bf",
            "technology": "#a78bfa", "location": "#4ade80",
            "event": "#f472b6", "concept": "#60a5fa", "date": "#fb923c",
        }

        with self.driver.session() as session:
            # ノード
            node_records = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.name AS name, e.type AS type,
                       count(r) AS degree
            """).data()

            nodes, type_counts = [], {}
            for r in node_records:
                etype = r["type"] or "concept"
                type_counts[etype] = type_counts.get(etype, 0) + 1
                nodes.append({
                    "id": r["name"], "label": r["name"], "type": etype,
                    "color": type_colors.get(etype, "#888"),
                    "degree": r["degree"],
                    "highlighted": r["name"] in highlighted,
                    "search_score": highlighted.get(r["name"], 0),
                })

            # エッジ
            edge_records = session.run("""
                MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                RETURN a.name AS source, b.name AS target,
                       r.relation AS relation
            """).data()

            edges = [{
                "source": r["source"], "target": r["target"],
                "relation": r["relation"] or "",
                "highlighted": r["source"] in highlighted and r["target"] in highlighted,
            } for r in edge_records]

            # 統計
            stats_r = session.run("""
                MATCH (e:Entity)
                WITH count(e) AS n_nodes
                OPTIONAL MATCH ()-[r:RELATES]->()
                WITH n_nodes, count(r) AS n_edges
                RETURN n_nodes, n_edges
            """).single()

            n_nodes = stats_r["n_nodes"] if stats_r else 0
            n_edges = stats_r["n_edges"] if stats_r else 0
            density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

            # 中心性TOP5（次数中心性）
            centrality_records = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e.name AS name, count(r) AS deg
                ORDER BY deg DESC LIMIT 5
                RETURN name, deg
            """).data()

            max_deg = centrality_records[0]["deg"] if centrality_records else 1
            centrality_top = [
                {"name": r["name"], "score": round(r["deg"] / max(max_deg, 1), 3)}
                for r in centrality_records
            ]

        return {
            "nodes": nodes, "edges": edges,
            "type_counts": type_counts, "type_colors": type_colors,
            "stats": {
                "nodes": n_nodes, "edges": n_edges,
                "density": round(density, 3), "components": 0,
                "centrality_top": centrality_top,
            },
        }

    def get_stats(self) -> dict:
        """グラフ統計"""
        with self.driver.session() as session:
            r = session.run("""
                OPTIONAL MATCH (c:Chunk) WITH count(c) AS chunks
                OPTIONAL MATCH (e:Entity) WITH chunks, count(e) AS entities
                OPTIONAL MATCH ()-[r:RELATES]->() WITH chunks, entities, count(r) AS relations
                OPTIONAL MATCH ()-[r2:NEXT_CHUNK|SHARED_ENTITY]-()
                RETURN chunks, entities, relations, count(r2) AS chunk_edges
            """).single()
        return {
            "nodes": r["entities"] if r else 0,
            "edges": r["relations"] if r else 0,
            "communities": 0,
            "chunks": r["chunks"] if r else 0,
        }

    def reset_entities(self):
        """エンティティグラフをリセット"""
        with self.driver.session() as session:
            session.run("MATCH (e:Entity) DETACH DELETE e")

    def reextract_entities(self) -> dict:
        """既存チャンクから再抽出"""
        self.reset_entities()
        with self.driver.session() as session:
            docs = session.run("""
                MATCH (c:Chunk)
                RETURN c.doc_id AS doc_id, collect(c.text) AS texts
                ORDER BY c.doc_id
            """).data()

        for doc in docs:
            full_text = "\n".join(doc["texts"])
            self._extract_and_add_entities(full_text)

        stats = self.get_stats()
        return {
            "status": "re-extracted", "documents": len(docs),
            "entities": stats["nodes"], "relations": stats["edges"],
        }

    # ========================================
    # 内部ユーティリティ
    # ========================================
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embeddings.embed_query(text)
        return np.array(vec, dtype=np.float32)

    def _split_text(self, text, chunk_size=500, overlap=50):
        if len(text) <= chunk_size:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for sep in ["。\n", "。", "\n\n", "\n", "、"]:
                    pos = text[start:end].rfind(sep)
                    if pos > chunk_size // 2:
                        end = start + pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _build_shared_edges(self, session, doc_id: str):
        """同一ドキュメント内の共通キーワードエッジ"""
        import re
        records = session.run("""
            MATCH (c:Chunk {doc_id: $doc_id})
            RETURN c.chunk_id AS id, c.text AS text
            ORDER BY c.seq
        """, doc_id=doc_id).data()

        for i, c1 in enumerate(records):
            w1 = set(re.findall(r"[\u4e00-\u9fff]{2,}", c1["text"]))
            for c2 in records[i + 2:]:
                w2 = set(re.findall(r"[\u4e00-\u9fff]{2,}", c2["text"]))
                if len(w1 & w2) >= 2:
                    session.run("""
                        MATCH (a:Chunk {chunk_id: $a}), (b:Chunk {chunk_id: $b})
                        MERGE (a)-[:SHARED_ENTITY]->(b)
                    """, a=c1["id"], b=c2["id"])


# シングルトン
_store: Optional[Neo4jStore] = None


def get_neo4j_store() -> Neo4jStore:
    global _store
    if _store is None:
        _store = Neo4jStore()
        _store.init_schema()
    return _store
