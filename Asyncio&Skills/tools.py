import os
import json
import inspect
from typing import Any, List, Dict, Optional # 引入复杂的类型提示
from pydantic import create_model, Field     # 引入 Pydantic 神器
from config import tavily


# --- 技能包元数据（描述各领域的作用，专门喂给 Router 看的） ---
CATEGORY_METADATA = {
    "office": "涉及读取/写入本地文件、代码文档、收发邮件、办公自动化操作。",
    "gamedev": "涉及游戏开发、引擎崩溃报错日志分析、剧情对话树生成、游戏数值平衡。"
    # base 技能（搜索和提交）是底层被动技能，不需要让 Router 知道，直接默认加载
}


# --- 将原本单一的字典，升级为分类存储的技能注册表 ---
SKILL_REGISTRY = {
    "base": {"tools": {}, "schemas": []},
    "office": {"tools": {}, "schemas":[]},
    "gamedev": {"tools": {}, "schemas":[]}
}

def register_tool(category="base"):
    """
    工业级工具注册器：自动将复杂的 Python Type Hints (如 List[str], Dict) 
    无损转化为 OpenAI 支持的严格 JSON Schema。
    """
    def decorator(func):
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # --- 动态收集 Pydantic 字段配置 ---
        fields = {}
        for name, param in sig.parameters.items():
            # --- 上下文穿透 ---
            # 如果参数名叫 agent_context，直接跳过，绝不暴露给大模型
            if name == "agent_context":
                continue

            # 获取参数的类型注解，如果没写类型，默认视为 Any (任何类型)
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # 判断是否有默认值。如果没有默认值，在 Pydantic 中用 ... 表示必填 (Required)
            if param.default == inspect.Parameter.empty:
                default_val = ...
            else:
                default_val = param.default

            # 将每个参数组装为 Pydantic 所需的 Field
            fields[name] = (annotation, Field(default=default_val, description=f"参数 {name}"))
            
        # 运行时“凭空”捏造一个 Pydantic 数据模型类 (Metaprogramming)
        # 例如定义了 def calc(a: int), 这里就会在内存里生成一个 class calc_Params(BaseModel): a: int
        pydantic_model = create_model(f"{func.__name__}_Params", **fields)
        
        # 直接让 Pydantic 交出 JSON Schema
        model_schema = pydantic_model.model_json_schema()

        # 移除 Pydantic 自动生成的 title 字段（为了让传给 OpenAI 的 JSON 更干净省 Token）
        model_schema.pop("title", None)

        # 组装成 OpenAI 标准 Function Calling 格式
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.strip().replace("\n", " "), # 将多行注释压缩
                "parameters": model_schema  # 直接把 Pydantic 生成的完美字典塞进来
            }
        }
        
        # 将工具存入对应的分类抽屉中
        if category not in SKILL_REGISTRY:
            SKILL_REGISTRY[category] = {"tools": {}, "schemas": []}
        
        SKILL_REGISTRY[category]["tools"][func.__name__] = func
        SKILL_REGISTRY[category]["schemas"].append(schema)
        return func
    return decorator


# Base Skill

@register_tool(category="base")
def google_search(query: str) -> str:
    """
    当用户询问实时信息、新闻、历史事实、特定人物或需要查阅互联网资料时使用。
    必须遵循'全要素'原则，输入具体的搜索查询语句（例如：'特斯拉(Tesla)最新股价行情'）。
    """
    try:
        response = tavily.search(query=query, search_depth="advanced", max_results=3)
        results =[f"来源: {r['url']}\n内容: {r['content']}" for r in response['results']]
        return "\n\n".join(results) if results else "未找到相关搜索结果。"
    except Exception as e:
        return f"搜索过程中发生错误: {str(e)}"

# 大模型使用的“提交答案”工具
@register_tool(category="base")
def submit_final_answer(answer: str) -> str:
    """
    当且仅当你确信已经完美解决用户问题时，调用此工具将最终答案提交给用户。
    注意：排版要精美，使用 Markdown 格式。
    """
    # 这个函数本身不需要真实执行逻辑，它的参数会被 engine.py 拦截并触发 Reflection
    return "已提交审核"


# Office Skill

@register_tool(category="office")
async def read_local_file(file_path: str) -> str:
    """读取本地计算机上的文本文件（如 .txt, .md, .py）。传入绝对或相对路径。"""
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在。"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 防止文件过大撑爆上下文，截断前 10000 个字符
        return content[:10000] + ("\n...(已截断)" if len(content) > 10000 else "")
    except Exception as e:
        return f"读取文件失败: {e}"

# 重写 write_local_file 工具，让它自动在特定 session_id 的专属文件夹里保存文件
@register_tool(category="office")
async def write_local_file(file_name: str, content: str, agent_context=None) -> str:
    """
    将内容写入或覆盖到本地文件中。如果文件不存在会自动创建。
    """
    try:
        # 上下文穿透作用：大模型只传了 file_name 和 content
        # 但我们在底层拿到了 Agent 实例，从而可以读取它的 session_id！
        session_id = agent_context.session_id if agent_context else "default_session"

        # 为每个对话创建一个专属文件夹
        save_dir = os.path.join("workspace", session_id)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功！内容已隔离保存至您的专属工作区: {file_path}"
    except Exception as e:
        return f"写入文件失败: {e}"

@register_tool(category="office")
async def send_mock_email(to_address: str, subject: str, body: str) -> str:
    """发送工作邮件给指定联系人。"""
    # 这里我们用 mock 模拟，如果你有真实 SMTP 需求可以替换
    return f"【邮件发送成功】收件人: {to_address} | 主题: {subject}\n系统提示：已成功投递。"


# GameDev Skill

@register_tool(category="gamedev")
async def analyze_engine_log(log_snippet: str) -> str:
    """
    当 Unity 或 Unreal 引擎崩溃时，分析报错日志切片。
    输入一段 Exception 日志，返回可能导致崩溃的 C#/C++ 模块定位。
    """
    # 模拟日志分析逻辑
    if "NullReferenceException" in log_snippet:
        return "诊断结果：空引用异常。建议检查 Awake 或 Start 阶段的 GameObject 绑定是否丢失。"
    elif "Access Violation" in log_snippet:
        return "诊断结果：C++ 内存越界或野指针。请检查最近修改的 Unmanaged 内存分配逻辑。"
    return "诊断结果：未知报错。请尝试使用 google_search 工具查找该 Error Code。"

@register_tool(category="gamedev")
async def generate_dialogue_json(npc_name: str, topic: str) -> str:
    """
    根据剧情设定，为指定 NPC 生成合法的对话树结构（直接输出供引擎读取的 JSON 字符串格式）。
    """
    mock_tree = {
        "NPC": npc_name,
        "Nodes":[
            {"id": 1, "text": f"关于{topic}，其实我知道的不多...", "options":[
                {"choice": "继续追问", "next_node": 2},
                {"choice": "离开", "next_node": -1}
            ]}
        ]
    }
    return json.dumps(mock_tree, ensure_ascii=False, indent=2)

# Pydantic 演示用例：极其复杂的嵌套参数，Pydantic 也能一秒解析
@register_tool(category="gamedev")
async def batch_update_monsters(
    scene_id: str, 
    monsters_data: List[Dict[str, int]], # 魔法在这里！List 里面套 Dict，Dict 里面套字符串和数字
    is_boss_level: Optional[bool] = False # 甚至支持 Optional (非必填项)
) -> str:
    """批量更新当前场景下怪物的血量和攻击力。"""
    # 真实执行逻辑...
    return f"已更新场景 {scene_id} 中的 {len(monsters_data)} 只怪物。"
