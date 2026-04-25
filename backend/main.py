"""
FastAPI + LangGraph バックエンド
React → Apache(リバプロ) → ここ → Ollama
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio

from graph_agent import build_agent, AgentState

app = FastAPI(title="GraphRAG Local LLM", version="1.0.0")

# CORS設定（開発時はReact devserver許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraphエージェントをアプリ起動時に構築
agent = build_agent()


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
    """通常のチャットエンドポイント（非ストリーミング）"""
    try:
        initial_state: AgentState = {
            "question": req.message,
            "chat_history": req.history,
            "search_results": [],
            "community_info": "",
            "answer": "",
            "quality_score": 0.0,
            "retry_count": 0,
            "search_query": "",
        }

        result = await asyncio.to_thread(agent.invoke, initial_state)

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("search_results", []),
            quality_score=result.get("quality_score", 0.0),
            retry_count=result.get("retry_count", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """ストリーミング対応エンドポイント（SSE）"""

    async def event_generator():
        initial_state: AgentState = {
            "question": req.message,
            "chat_history": req.history,
            "search_results": [],
            "community_info": "",
            "answer": "",
            "quality_score": 0.0,
            "retry_count": 0,
            "search_query": "",
        }

        # LangGraphの各ノード実行をストリーミング
        for event in agent.stream(initial_state):
            for node_name, node_output in event.items():
                payload = {
                    "node": node_name,
                    "data": _serialize(node_output),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _serialize(obj: dict) -> dict:
    """numpy配列などをJSON変換可能な形式に変換"""
    result = {}
    for k, v in obj.items():
        if hasattr(v, "tolist"):
            result[k] = v.tolist()
        elif isinstance(v, list) and v and hasattr(v[0], "tolist"):
            result[k] = [x.tolist() for x in v]
        else:
            result[k] = v
    return result


# --- ドキュメント管理 ---

class DocumentUpload(BaseModel):
    text: str
    metadata: dict = {}


@app.post("/api/documents")
async def add_document(doc: DocumentUpload):
    """ドキュメントをGraphRAGに追加"""
    from graph_rag import get_retriever
    retriever = get_retriever()
    doc_id = retriever.add_document(doc.text, doc.metadata)
    return {"doc_id": doc_id, "status": "indexed"}


@app.post("/api/graph/reextract")
async def reextract_entities():
    """既存チャンクからエンティティを再抽出（プロンプト変更後に使用）"""
    from graph_rag import get_retriever
    import networkx as nx_lib
    retriever = get_retriever()

    # エンティティグラフをリセット
    retriever.entity_graph = nx_lib.DiGraph()

    # 全チャンクのテキストを結合してドキュメント単位で再抽出
    doc_texts = {}
    for chunk in retriever.chunks:
        doc_id = chunk.get("doc_id", "unknown")
        if doc_id not in doc_texts:
            doc_texts[doc_id] = []
        doc_texts[doc_id].append(chunk["text"])

    total_entities = 0
    total_relations = 0
    for doc_id, texts in doc_texts.items():
        full_text = "\n".join(texts)
        retriever._extract_and_add_entities(full_text)
        total_entities = retriever.entity_graph.number_of_nodes()
        total_relations = retriever.entity_graph.number_of_edges()

    retriever.save()

    return {
        "status": "re-extracted",
        "documents": len(doc_texts),
        "entities": total_entities,
        "relations": total_relations,
    }


@app.post("/api/graph/reset")
async def reset_graph():
    """エンティティグラフをリセット"""
    from graph_rag import get_retriever
    import networkx as nx_lib
    retriever = get_retriever()
    retriever.entity_graph = nx_lib.DiGraph()
    retriever.save()
    return {"status": "reset"}


@app.get("/api/graph/stats")
async def graph_stats():
    """知識グラフの統計情報"""
    from graph_rag import get_retriever
    retriever = get_retriever()
    g = retriever.graph
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "communities": len(retriever.communities),
        "chunks": len(retriever.chunks),
    }


@app.get("/api/graph/data")
async def graph_data():
    """エンティティグラフの可視化データ（ノード+エッジ+統計）"""
    from graph_rag import get_retriever
    import networkx as nx_lib
    retriever = get_retriever()
    g = retriever.entity_graph

    # エンティティタイプ → 色マッピング
    type_colors = {
        "person": "#ef4444",
        "organization": "#2dd4bf",
        "technology": "#a78bfa",
        "location": "#4ade80",
        "event": "#f472b6",
        "concept": "#60a5fa",
        "date": "#fb923c",
    }

    # ノード
    nodes = []
    type_counts = {}
    for node_id in g.nodes():
        attrs = g.nodes[node_id]
        etype = attrs.get("type", "concept")
        type_counts[etype] = type_counts.get(etype, 0) + 1

        nodes.append({
            "id": node_id,
            "label": node_id,
            "type": etype,
            "color": type_colors.get(etype, "#888"),
            "degree": g.degree(node_id),
        })

    # エッジ
    edges = []
    for u, v, data in g.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relation": data.get("relation", ""),
        })

    # グラフ統計
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = nx_lib.density(g) if n_nodes > 1 else 0

    # 連結成分（有向→無向に変換して計算）
    if n_nodes > 0:
        undirected = g.to_undirected()
        n_components = nx_lib.number_connected_components(undirected)
    else:
        n_components = 0

    # 次数中心性トップ5
    centrality_top = []
    if n_nodes > 0:
        centrality = nx_lib.degree_centrality(g)
        sorted_c = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        centrality_top = [
            {"name": name, "score": round(score, 3)}
            for name, score in sorted_c[:5]
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "type_counts": type_counts,
        "type_colors": type_colors,
        "stats": {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": round(density, 3),
            "components": n_components,
            "centrality_top": centrality_top,
        },
    }


@app.get("/api/health")
async def health():
    """ヘルスチェック"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ok",
        "ollama": ollama_ok,
    }