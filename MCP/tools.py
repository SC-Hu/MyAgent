import os
import json
import inspect
import asyncio
from typing import Any, List, Dict, Optional # 引入复杂的类型提示
from pydantic import create_model, Field     # 引入 Pydantic 神器
from config import tavily
from config import Config


# --- Toolkit 元数据：这是给 Router 看的“部门简介” ---
TOOLKIT_METADATA = {
    "office": "涉及读取/生成/写入本地文件(如 txt, md, word, doc 等)、代码文档、收发邮件、办公自动化操作。",
    "gamedev": "游戏开发管线：处理 Unity/Unreal 引擎日志、生成对话 JSON、调整怪物数值等游戏业务。",
    "system": "执行底层终端命令(Bash/CMD)、管理本地系统环境、运行脚本、安装软件包。该模块操作具有高风险。" # 新增
    # base 技能（搜索和提交）是底层被动技能，不需要让 Router 知道，直接默认加载
}


# --- 将原本单一的字典，升级为分类存储的技能注册表 ---
TOOLKIT_REGISTRY = {}

""" 注册表结构
{
    "toolkit name": {
        "description": "toolkit description",
        "tools": {
            "native::office::write_local_file": {
                "func": <function write_local_file at 0x...>,
                "requires_approval": False,
                "description": "在工作区生成文件。支持写入代码或文本内容。"
            }
        },
        "schemas": [
            { "type": "function", "function": { "name": "native::office::write_local_file", ... } }
        ]
    }
}
"""


def get_safe_path(target_path: str) -> str:
    """
    路径守卫：将用户/模型输入的路径转化为安全、受控的绝对路径。
    如果路径超出 workspace，抛出 PermissionError。
    """
    # 处理模型可能输入的相对路径
    # 即使模型输入 "../../etc/passwd"，abspath 也会将其还原为真实的物理路径
    absolute_target = os.path.abspath(os.path.join(Config.WORKSPACE_ROOT, target_path))
    
    # 核心校验：目标路径必须以 WORKSPACE_ROOT 开头
    if not absolute_target.startswith(Config.WORKSPACE_ROOT):
        raise PermissionError("超出权限：操作路径位于工作区外。")
    
    return absolute_target


def register_tool(toolkit: str, source="native", requires_approval=False):
    """
    全限定名工具注册器。
    生成 ID 格式：source::toolkit::toolname
    """
    def decorator(func):
        tool_name = func.__name__
        # 生成全限定名 ID
        tool_id = f"{source}_{toolkit}_{tool_name}"

        # Pydantic Schema 生成逻辑
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        fields = {}
        for name, param in sig.parameters.items():
            # 如果参数名叫 agent_context，直接跳过，绝不暴露给大模型
            if name == "agent_context":
                continue
            # 获取参数的类型注解，如果没写类型，默认视为 str
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
            # 判断是否有默认值。如果没有默认值，在 Pydantic 中用 ... 表示必填 (Required)
            if param.default == inspect.Parameter.empty:
                default_val = ...
            else:
                default_val = param.default
            # 将每个参数组装为 Pydantic 所需的 Field
            fields[name] = (annotation, Field(default=default_val, description=f"参数 {name}"))
            
        # 运行时“凭空”捏造一个 Pydantic 数据模型类 (Metaprogramming)
        # 例如定义了 def calc(a: int), 这里就会在内存里生成一个 class calc_Params(BaseModel): a: int
        pydantic_model = create_model(f"{tool_name}_Params", **fields)
        
        # 直接让 Pydantic 交出 JSON Schema
        model_schema = pydantic_model.model_json_schema()

        # 移除 Pydantic 自动生成的 title 字段（为了让传给 OpenAI 的 JSON 更干净省 Token）
        model_schema.pop("title", None)

        """ Pydantic 结构
        {
            "type": "object",
            "properties": {
                "file_name": {
                "type": "string",
                "description": "参数 file_name"
                },
                "content": {
                "type": "string",
                "description": "参数 content"
                }
            },
            "required": ["file_name", "content"],
            "additionalProperties": false
        }
        """

        # 组装成 OpenAI 标准 Function Calling 格式
        schema = {
            "type": "function",
            "function": {
                "name": tool_id, # 使用全限定名作为函数名
                "description": doc.strip().replace("\n", " "), # 将多行注释压缩
                "parameters": model_schema  # 直接把 Pydantic 生成的完美字典塞进来
            }
        }
        
        """ OpenAI 协议格式
        {
            "type": "function",
            "function": {
                "name": "native::office::write_local_file",
                "description": "在工作区生成文件。支持写入代码或文本内容。",
                "parameters": { ... 上面的 model_schema ... }
            }
        }
        """

        # 注册到工具箱
        if toolkit not in TOOLKIT_REGISTRY:
            TOOLKIT_REGISTRY[toolkit] = {
                "description": TOOLKIT_METADATA.get(toolkit, ""),
                "tools": {},
                "schemas": []
            }
        
        TOOLKIT_REGISTRY[toolkit]["tools"][tool_id] = {
            "func": func,
            "requires_approval": requires_approval, # 是否需要审批
            "description": inspect.getdoc(func).strip() # 用于 RAG 检索的文本
        }
        TOOLKIT_REGISTRY[toolkit]["schemas"].append(schema)

        return func
    return decorator


# System Skill

@register_tool(toolkit="system", requires_approval=True)
async def execute_bash(command: str, agent_context=None) -> str:
    """
    【高危】在本地 workspace 目录下执行终端命令。
    禁止使用 '..' 跳转路径。
    可执行安装依赖、运行程序、Git 操作等。
    """
    # 静态防御：禁止路径逃逸尝试
    if ".." in command:
        return "拒绝执行：检测到非法路径跳转 '..'。"

    # 针对 Windows 的必须优化：强制使用 UTF-8 编码
    import sys
    final_command = f"chcp 65001 > nul && {command}" if sys.platform == "win32" else command

    # 建立异步子进程
    try:
        process = await asyncio.create_subprocess_shell(
            final_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Config.WORKSPACE_ROOT # 强制锚定在 workspace 运行
        )

        # 等待执行结果（可以设置超时，防止死循环命令）
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            process.kill()
            return "命令执行超时（30秒限制），已强制停止。"
        
        # 智能解码：尝试 utf-8, 失败则尝试 gbk
        def smart_decode(data: bytes) -> str:
            if not data: return ""
            for encoding in ['utf-8', 'gbk']:
                try: return data.decode(encoding).strip()
                except UnicodeDecodeError: continue
            return data.decode('utf-8', errors='replace').strip()

        # 格式化输出
        result_out = smart_decode(stdout)
        result_err = smart_decode(stderr)

        # 构造返回给大模型的报告
        report = []
        if result_out:
            report.append(f"[标准输出]:\n{result_out}")
        if result_err:
            report.append(f"[错误/警告输出]:\n{result_err}")
        if not result_out and not result_err:
            report.append("命令已执行，但无任何控制台输出。")
            
        return "\n".join(report)

    except Exception as e:
        return f"子进程执行异常: {str(e)}"
    

# Base Skill

@register_tool(toolkit="base")
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
@register_tool(toolkit="base")
def submit_final_answer(answer: str) -> str:
    """
    当且仅当你确信已经完美解决用户问题时，调用此工具将最终答案提交给用户。
    注意：排版要精美，使用 Markdown 格式。
    """
    # 这个函数本身不需要真实执行逻辑，它的参数会被 engine.py 拦截并触发 Reflection
    return "已提交审核"


# Office Skill

@register_tool(toolkit="office")
async def read_local_file(file_path: str) -> str:
    """读取本地计算机上的文本文件（如 .txt, .md, .py）。传入绝对或相对路径。"""
    try:
        # 使用沙盒路径
        safe_path = get_safe_path(file_path)

        if not os.path.exists(safe_path):
            return f"错误：文件 {file_path} 不存在。"
        
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 防止文件过大撑爆上下文，截断前 10000 个字符
        return content[:10000] + ("\n...(已截断)" if len(content) > 10000 else "")
    
    except PermissionError as e:
        return f"权限拒绝：你只能读取 workspace 目录内的文件。"
    except Exception as e:
        return f"读取文件失败: {e}"

# 重写 write_local_file 工具，让它自动在特定 session_id 的专属文件夹里保存文件
@register_tool(toolkit="office")
async def write_local_file(file_name: str, content: str) -> str:
    """
    将内容写入或覆盖到本地文件中。如果文件不存在会自动创建。
    【重要指令】：如果用户要求生成 Word 文档，请将内容排版好（如使用 Markdown），
    并强制将 file_name 的后缀名命名为 .md 或 .doc 返回。
    """
    try:
        # 使用沙盒路径
        safe_path = get_safe_path(file_name)
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功！文件已保存至: {os.path.relpath(safe_path, Config.WORKSPACE_ROOT)}"
    except PermissionError as e:
        return str(e) # 返回给模型：超出权限
    except Exception as e:
        return f"写入失败: {e}"

@register_tool(toolkit="office")
async def send_mock_email(to_address: str, subject: str, body: str) -> str:
    """发送工作邮件给指定联系人。"""
    # 这里我们用 mock 模拟，如果你有真实 SMTP 需求可以替换
    return f"【邮件发送成功】收件人: {to_address} | 主题: {subject}\n系统提示：已成功投递。"


# GameDev Skill

@register_tool(toolkit="gamedev")
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

@register_tool(toolkit="gamedev")
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
@register_tool(toolkit="gamedev")
async def batch_update_monsters(
    scene_id: str, 
    monsters_data: List[Dict[str, int]], # 魔法在这里！List 里面套 Dict，Dict 里面套字符串和数字
    is_boss_level: Optional[bool] = False # 甚至支持 Optional (非必填项)
) -> str:
    """批量更新当前场景下怪物的血量和攻击力。"""
    # 真实执行逻辑...
    return f"已更新场景 {scene_id} 中的 {len(monsters_data)} 只怪物。"
