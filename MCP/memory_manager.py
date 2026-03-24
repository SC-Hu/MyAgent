# --- memory_manager.py ---
import os
import uuid
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from config import logger
from dotenv import load_dotenv
import openai


load_dotenv()


# 配置自定义 Embedding 函数，对接 API
# 这里模拟一个符合 ChromaDB 标准的 Embedding 函数
class CustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 这里直接使用在 .env 中定义的 EBD 系列环境变量
        client = openai.OpenAI(
            api_key=os.getenv("EBD_API_KEY"),
            base_url=os.getenv("EBD_BASE_URL")
        )

        # 服务商限制：单次 API 请求最多 32 个文本块
        MAX_BATCH_SIZE = 32
        all_embeddings = []

        # 核心修复：把大数组切分成多个最多 32 长度的小批次，循环发送
        for i in range(0, len(input), MAX_BATCH_SIZE):
            batch_input = input[i : i + MAX_BATCH_SIZE]
            
            try:
                response = client.embeddings.create(
                    input=batch_input,
                    model=os.getenv("EBD_MODEL_NAME")
                )
                # 将这一批次的向量结果追加到总列表中
                all_embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                logger.error(f"Embedding API 第 {i} 批次请求失败: {e}")
                # 生产环境建议抛出异常或重试，这里直接抛出让上层捕获
                raise e 
                
        return all_embeddings


class MemoryManager:
    def __init__(self):
        # 初始化 ChromaDB 持久化存储
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.emb_fn = CustomEmbeddingFunction()
        
        # --- 1. 对话记忆集合 (原有功能) ---
        self.chat_collection = self.client.get_or_create_collection(
            name="agent_long_term_memory",
            embedding_function=self.emb_fn
        )

        # --- 2. 工具索引集合 (新增功能) ---
        self.tool_collection = self.client.get_or_create_collection(
            name="agent_tools_index",
            embedding_function=self.emb_fn
        )

        self.user_id = "master_user"

    def save_facts(self, text: str):
        """将事实存入 ChromaDB (附带防 512 Token 超限的截断分块功能)"""
        # 简单暴力的切分逻辑：假设 1 Token ≈ 1.5 到 2 个中文字符
        # 这里设置为每 2000 个字符切一块，绝对安全不会触发 512 Token 报错
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        if not chunks: return

        try:
            # 批处理性能优化
            # 一次性生成所有的 ID 和 Metadata
            mem_ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"user_id": self.user_id} for _ in chunks]

            # 把数组直接传给 ChromaDB！底层 Embedding 函数会把整个数组一次性发给 API
            self.chat_collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=mem_ids
            )
            logger.info(f"🧠 [长期记忆] 事实已分成 {len(chunks)} 个碎片，沉淀至向量库。")
        except Exception as e:
            logger.error(f"长期记忆写入失败: {e}")

    def retrieve(self, query: str, limit: int = 3, threshold: float = 1.3) -> str:
        """检索：带有严格阈值过滤的高级捞取"""
        try:
            results = self.chat_collection.query(
                query_texts=[query],
                n_results=limit,
                where={"user_id": self.user_id} # 物理隔离：只捞当前用户的数据
            )
            docs = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            valid_docs = []

            # 核心控制逻辑：双重循环遍历对比
            for doc, distance in zip(docs, distances):
                # 因为 Chroma 默认用 L2 距离，所以距离越小越好，这里用 < 号
                # 如果你的 Chroma 配置成了 Cosine，这里就要改成 distance > threshold
                if distance < threshold:
                    valid_docs.append(doc)
                    logger.debug(f"放行: 距离 {distance:.4f} < 阈值 {threshold}")
                else:
                    logger.warning(f"拦截: 距离 {distance:.4f} >= 阈值 {threshold} (文本已被丢弃)")

            return "\n".join([f"- {doc}" for doc in valid_docs]) if valid_docs else ""
        except Exception as e:
            logger.error(f"长期记忆检索失败: {e}")
            return ""
    

    def index_all_tools(self, toolkit_registry: dict):
        """
        [系统级任务] 将所有注册的工具描述存入向量库
        通常在 main.py 启动时调用一次
        """
        ids, documents, metadatas = [], [], []
        
        for tk_name, tk_data in toolkit_registry.items():
            for tool_id, tool_info in tk_data["tools"].items():
                ids.append(tool_id) # 这里是 native::office::write_local_file
                # 我们通过工具的 Docstring 进行检索
                documents.append(tool_info["description"])
                # 存入元数据，方便后面反查它属于哪个工具箱
                metadatas.append({"toolkit": tk_name})
        
        if ids:
            # 使用 upsert，如果 ID 已存在则更新描述
            self.tool_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(f"🛠️ [工具索引] 已同步 {len(ids)} 个工具至 RAG 引擎。")

    def search_toolkits(self, query: str, active_domains: list, limit: int = 3) -> list:
        """
        [决策级任务] 在 Router 选定的领域内，寻找最匹配的 Toolkit 名字
        """
        try:
            results = self.tool_collection.query(
                query_texts=[query],
                n_results=limit,
                # 关键：这里使用了元数据过滤，只在 Router 划定的范围内搜
                where={"toolkit": {"$in": active_domains}} 
            )
            
            # 提取元数据中的 toolkit 名字并去重
            matched_toolkits = []
            if results.get("metadatas"):
                for meta in results["metadatas"][0]:
                    if meta["toolkit"] not in matched_toolkits:
                        matched_toolkits.append(meta["toolkit"])
            
            return matched_toolkits
        except Exception as e:
            logger.error(f"工具检索失败: {e}")
            return active_domains # 失败则保底返回 Router 选的所有领域

# 实例化全局单例
long_term_memory = MemoryManager()