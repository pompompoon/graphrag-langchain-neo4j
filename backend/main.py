"""
FastAPI バックエンド（ストレージ切り替え対応）
STORAGE=file → graph_rag.py（ファイルベース）
STORAGE=neo4j → neo4j_store.py（Neo4j）
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio

from config import STORAGE, get_provider_info
from graph_agent import build_agent, AgentState

app = FastAPI(title="GraphRAG Local LLM", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

agent = build_agent()


# ============================================================
# ストレージ抽象化
# ============================================================
def _store():
    """STORAGE設定に応じたストアを返す"""
    if STORAGE == "neo4j":
        from neo4j_store import get_neo4j_store
        return get_neo4j_store()
    else:
        from graph_rag import get_retriever
        return get_retriever()


# ============================================================
# チャット
# ============================================================
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    quality_score: float
    retry_count: int


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        initial_state: AgentState = {
            "question": req.message, "chat_history": req.history,
            "search_results": [], "community_info": "", "answer": "",
            "quality_score": 0.0, "retry_count": 0, "search_query": "",
        }
        result = await asyncio.to_thread(agent.invoke, initial_state)
        return ChatResponse(
            answer=result["answer"], sources=result.get("search_results", []),
            quality_score=result.get("quality_score", 0.0),
            retry_count=result.get("retry_count", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        initial_state: AgentState = {
            "question": req.message, "chat_history": req.history,
            "search_results": [], "community_info": "", "answer": "",
            "quality_score": 0.0, "retry_count": 0, "search_query": "",
        }
        for event in agent.stream(initial_state):
            for node_name, node_output in event.items():
                payload = {"node": node_name, "data": _serialize(node_output)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            store = _store()
            matched = store.search_entities(req.message, top_k=5)
            entity_payload = {
                "node": "entity_match",
                "data": {"matched_entities": [
                    {"name": e["name"], "score": e["score"], "type": e["type"]}
                    for e in matched if e["score"] > 0.3
                ]},
            }
            yield f"data: {json.dumps(entity_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[EntitySearch] Error: {e}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _serialize(obj: dict) -> dict:
    result = {}
    for k, v in obj.items():
        if hasattr(v, "tolist"):
            result[k] = v.tolist()
        elif isinstance(v, list) and v and hasattr(v[0], "tolist"):
            result[k] = [x.tolist() for x in v]
        else:
            result[k] = v
    return result


# ============================================================
# エンティティ検索
# ============================================================
class EntitySearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/api/entities/search")
async def search_entities_api(req: EntitySearchRequest):
    store = _store()
    results = store.search_entities(req.query, top_k=req.top_k)
    return {"entities": results, "query": req.query}


# ============================================================
# ドキュメント管理
# ============================================================
class DocumentUpload(BaseModel):
    text: str
    metadata: dict = {}

@app.post("/api/documents")
async def add_document(doc: DocumentUpload):
    store = _store()
    doc_id = store.add_document(doc.text, doc.metadata)
    return {"doc_id": doc_id, "status": "indexed"}


@app.post("/api/graph/reextract")
async def reextract_entities():
    store = _store()
    if STORAGE == "neo4j":
        return store.reextract_entities()
    else:
        import networkx as nx_lib
        store.entity_graph = nx_lib.DiGraph()
        doc_texts = {}
        for chunk in store.chunks:
            doc_id = chunk.get("doc_id", "unknown")
            if doc_id not in doc_texts:
                doc_texts[doc_id] = []
            doc_texts[doc_id].append(chunk["text"])
        for texts in doc_texts.values():
            store._extract_and_add_entities("\n".join(texts))
        store.save()
        return {"status": "re-extracted",
                "entities": store.entity_graph.number_of_nodes(),
                "relations": store.entity_graph.number_of_edges()}


@app.post("/api/graph/reset")
async def reset_graph():
    store = _store()
    if STORAGE == "neo4j":
        store.reset_entities()
    else:
        import networkx as nx_lib
        store.entity_graph = nx_lib.DiGraph()
        store.save()
    return {"status": "reset"}


# ============================================================
# グラフ統計・可視化
# ============================================================
@app.get("/api/graph/stats")
async def graph_stats():
    store = _store()
    if STORAGE == "neo4j":
        return store.get_stats()
    else:
        g = store.graph
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                "communities": len(store.communities), "chunks": len(store.chunks)}


@app.get("/api/graph/data")
async def graph_data(query: str = ""):
    store = _store()
    if STORAGE == "neo4j":
        return store.get_graph_data(query)
    else:
        # ファイル版のグラフデータ取得（既存コードと同じ）
        import networkx as nx_lib
        g = store.entity_graph
        highlighted_ids, search_scores = set(), {}
        if query.strip() and len(store.entity_id_map) > 0:
            for er in store.search_entities(query, top_k=10):
                if er["score"] > 0.3:
                    highlighted_ids.add(er["name"])
                    search_scores[er["name"]] = er["score"]
                    for nb in er.get("neighbors", []):
                        if nb["name"] not in highlighted_ids:
                            highlighted_ids.add(nb["name"])
                            search_scores[nb["name"]] = er["score"] * 0.5

        type_colors = {
            "person": "#ef4444", "organization": "#2dd4bf",
            "technology": "#a78bfa", "location": "#4ade80",
            "event": "#f472b6", "concept": "#60a5fa", "date": "#fb923c"}

        nodes, type_counts = [], {}
        for nid in g.nodes():
            attrs = g.nodes[nid]
            etype = attrs.get("type", "concept")
            type_counts[etype] = type_counts.get(etype, 0) + 1
            nodes.append({"id": nid, "label": nid, "type": etype,
                          "color": type_colors.get(etype, "#888"),
                          "degree": g.degree(nid),
                          "highlighted": nid in highlighted_ids,
                          "search_score": search_scores.get(nid, 0)})

        edges = [{"source": u, "target": v, "relation": d.get("relation", ""),
                  "highlighted": u in highlighted_ids and v in highlighted_ids}
                 for u, v, d in g.edges(data=True)]

        n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
        density = nx_lib.density(g) if n_nodes > 1 else 0
        n_comp = nx_lib.number_connected_components(g.to_undirected()) if n_nodes > 0 else 0
        cent_top = []
        if n_nodes > 0:
            cent = nx_lib.degree_centrality(g)
            cent_top = [{"name": n, "score": round(s, 3)}
                        for n, s in sorted(cent.items(), key=lambda x: x[1], reverse=True)[:5]]

        return {"nodes": nodes, "edges": edges,
                "type_counts": type_counts, "type_colors": type_colors,
                "stats": {"nodes": n_nodes, "edges": n_edges,
                          "density": round(density, 3), "components": n_comp,
                          "centrality_top": cent_top}}


@app.get("/api/health")
async def health():
    import httpx
    info = get_provider_info()
    ollama_ok = False
    if info["provider"] == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:11434/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            pass

    neo4j_ok = False
    if STORAGE == "neo4j":
        try:
            store = _store()
            store.driver.verify_connectivity()
            neo4j_ok = True
        except Exception:
            pass

    return {"status": "ok", "ollama": ollama_ok, "neo4j": neo4j_ok, **info}
