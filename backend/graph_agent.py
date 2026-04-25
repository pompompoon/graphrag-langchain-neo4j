"""
LangGraph エージェント定義
ノード: クエリ分析 → GraphRAG検索 → 回答生成 → 品質チェック → (ループ or 完了)
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph_rag import get_retriever

# --- 設定 ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
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
    """LangGraphエージェントを構築してコンパイル済みグラフを返す"""

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )

    # ========================================
    # ノード1: クエリ分析・最適化
    # ========================================
    def analyze_query(state: AgentState) -> dict:
        """ユーザーの質問を分析し、GraphRAG検索に適したクエリに変換"""
        retry = state.get("retry_count", 0)

        if retry == 0:
            # 初回: 質問からキーワード抽出
            prompt = ChatPromptTemplate.from_template(
                "以下の質問を知識グラフ検索に最適な短いキーワード（日本語）に変換してください。\n"
                "キーワードのみをスペース区切りで出力してください。\n\n"
                "質問: {question}"
            )
            chain = prompt | llm | StrOutputParser()
            search_query = chain.invoke({"question": state["question"]})
        else:
            # リトライ: 前回の検索結果と回答を踏まえてクエリを改善
            prompt = ChatPromptTemplate.from_template(
                "以下の質問に対する検索結果が不十分でした。\n"
                "別の角度からの検索キーワードを生成してください。\n\n"
                "質問: {question}\n"
                "前回の検索クエリ: {prev_query}\n"
                "前回の回答: {prev_answer}\n\n"
                "新しいキーワードのみ出力:"
            )
            chain = prompt | llm | StrOutputParser()
            search_query = chain.invoke({
                "question": state["question"],
                "prev_query": state.get("search_query", ""),
                "prev_answer": state.get("answer", ""),
            })

        return {"search_query": search_query.strip()}

    # ========================================
    # ノード2: GraphRAG検索
    # ========================================
    def graph_rag_search(state: AgentState) -> dict:
        """NetworkX + hnswlib でグラフベース検索"""
        retriever = get_retriever()
        query = state.get("search_query", state["question"])

        # ベクトル検索 + グラフ近傍展開
        results = retriever.search(query, top_k=5)

        # コミュニティ要約を集約
        communities = set(r.get("community", "") for r in results if r.get("community"))
        community_info = "\n".join(
            [retriever.get_community_summary(c) for c in communities]
        )

        return {
            "search_results": results,
            "community_info": community_info,
        }

    # ========================================
    # ノード3: 回答生成
    # ========================================
    def generate_answer(state: AgentState) -> dict:
        """検索結果を踏まえてLLMで回答を生成"""
        context = "\n".join(
            [
                f"[{r.get('community', '不明')}] (スコア: {r.get('score', 0):.2f}) {r['text']}"
                for r in state["search_results"]
            ]
        )

        # チャット履歴をフォーマット
        history_text = ""
        if state.get("chat_history"):
            history_text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in state["chat_history"][-6:]]
            )

        prompt = ChatPromptTemplate.from_template(
            "あなたは知識グラフを活用するAIアシスタントです。\n"
            "検索結果とコミュニティ情報をもとに、正確かつ詳細に回答してください。\n"
            "情報が不足している場合はその旨を正直に伝えてください。\n\n"
            "## 会話履歴\n{history}\n\n"
            "## 検索結果（GraphRAGより）\n{context}\n\n"
            "## コミュニティ情報\n{community_info}\n\n"
            "## ユーザーの質問\n{question}\n\n"
            "回答:"
        )

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "history": history_text or "（なし）",
            "context": context or "（検索結果なし）",
            "community_info": state.get("community_info", "（なし）"),
            "question": state["question"],
        })

        return {"answer": answer}

    # ========================================
    # ノード4: 品質チェック
    # ========================================
    def check_quality(state: AgentState) -> dict:
        """LLMが自分の回答を自己評価"""
        prompt = ChatPromptTemplate.from_template(
            "以下の質問と回答を評価してください。\n\n"
            "質問: {question}\n"
            "回答: {answer}\n\n"
            "評価基準:\n"
            "- 質問に正確に答えているか\n"
            "- 具体的で有用な情報を含んでいるか\n"
            "- 「情報がない」と言いつつ実際は答えられていないか\n\n"
            "0.0〜1.0の数値のみ出力してください:"
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

    # ========================================
    # 条件分岐: リトライ判定
    # ========================================
    def should_retry(state: AgentState) -> Literal["retry", "done"]:
        if state.get("quality_score", 0) >= QUALITY_THRESHOLD:
            return "done"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "done"
        return "retry"

    # ========================================
    # グラフ構築
    # ========================================
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search", graph_rag_search)
    graph.add_node("generate", generate_answer)
    graph.add_node("check_quality", check_quality)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "check_quality")

    graph.add_conditional_edges(
        "check_quality",
        should_retry,
        {
            "retry": "analyze_query",
            "done": END,
        },
    )

    return graph.compile()
