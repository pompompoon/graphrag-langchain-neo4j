"""
LangGraph エージェント（LangChain完全統合版）
config.py経由でLLMを取得 → Ollama/Geminiの切り替えが自動
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_chat_llm

MAX_RETRIES = 2
QUALITY_THRESHOLD = 0.6


class AgentState(TypedDict):
    question: str
    chat_history: list[dict]
    search_results: list[dict]
    community_info: str
    answer: str
    quality_score: float
    retry_count: int
    search_query: str


def build_agent():
    """LangGraphエージェントを構築"""

    # config.py から LLM を取得（Ollama or Gemini）
    llm = get_chat_llm(temperature=0.3)

    def analyze_query(state: AgentState) -> dict:
        retry = state.get("retry_count", 0)
        if retry == 0:
            prompt = ChatPromptTemplate.from_template(
                "以下の質問を知識グラフ検索に最適な短いキーワード（日本語）に変換してください。\n"
                "キーワードのみをスペース区切りで出力してください。\n\n"
                "質問: {question}"
            )
            chain = prompt | llm | StrOutputParser()
            search_query = chain.invoke({"question": state["question"]})
        else:
            prompt = ChatPromptTemplate.from_template(
                "以下の質問に対する検索結果が不十分でした。\n"
                "別の角度からの検索キーワードを生成してください。\n\n"
                "質問: {question}\n前回のクエリ: {prev_query}\n\n"
                "新しいキーワードのみ出力:"
            )
            chain = prompt | llm | StrOutputParser()
            search_query = chain.invoke({
                "question": state["question"],
                "prev_query": state.get("search_query", ""),
            })
        return {"search_query": search_query.strip()}

    def graph_rag_search(state: AgentState) -> dict:
        from config import STORAGE
        if STORAGE == "neo4j":
            from neo4j_store import get_neo4j_store
            store = get_neo4j_store()
        else:
            from graph_rag import get_retriever
            store = get_retriever()

        query = state.get("search_query", state["question"])
        results = store.search(query, top_k=5)
        communities = set(r.get("community", "") for r in results if r.get("community"))
        community_info = "\n".join(
            [store.get_community_summary(c) for c in communities]
        )
        return {"search_results": results, "community_info": community_info}

    def generate_answer(state: AgentState) -> dict:
        context = "\n".join([
            f"[{r.get('community', '不明')}] (スコア: {r.get('score', 0):.2f}) {r['text']}"
            for r in state["search_results"]
        ])
        history_text = ""
        if state.get("chat_history"):
            history_text = "\n".join([
                f"{m['role']}: {m['content']}" for m in state["chat_history"][-6:]
            ])

        prompt = ChatPromptTemplate.from_template(
            "あなたは知識グラフを活用するAIアシスタントです。\n"
            "検索結果をもとに正確に回答してください。\n"
            "情報が不足している場合はその旨を伝えてください。\n\n"
            "## 会話履歴\n{history}\n\n"
            "## 検索結果（GraphRAGより）\n{context}\n\n"
            "## コミュニティ情報\n{community_info}\n\n"
            "## 質問\n{question}\n\n回答:"
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "history": history_text or "（なし）",
            "context": context or "（検索結果なし）",
            "community_info": state.get("community_info", "（なし）"),
            "question": state["question"],
        })
        return {"answer": answer}

    def check_quality(state: AgentState) -> dict:
        prompt = ChatPromptTemplate.from_template(
            "質問: {question}\n回答: {answer}\n\n"
            "この回答の品質を0.0〜1.0で評価してください。数値のみ出力:"
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
        })
        try:
            score = float(raw.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            score = 0.5
        return {
            "quality_score": score,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def should_retry(state: AgentState) -> Literal["retry", "done"]:
        if state.get("quality_score", 0) >= QUALITY_THRESHOLD:
            return "done"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "done"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search", graph_rag_search)
    graph.add_node("generate", generate_answer)
    graph.add_node("check_quality", check_quality)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "check_quality")
    graph.add_conditional_edges("check_quality", should_retry, {
        "retry": "analyze_query", "done": END,
    })

    return graph.compile()
