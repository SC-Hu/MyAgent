import re
import inspect
from config import tavily

# 自动收集工具字典和 Schema 列表
TOOL_MAP = {}
TOOLS_SCHEMA =[]

def register_tool(func):
    """
    自动解析 Python 函数签名，生成 OpenAI 标准的 JSON Schema
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    properties = {}
    required =[]
    
    for name, param in sig.parameters.items():
        # 将 Python 类型映射为 JSON Schema 类型
        param_type = "string"
        if param.annotation == int: param_type = "integer"
        elif param.annotation == float: param_type = "number"
        elif param.annotation == bool: param_type = "boolean"
        elif param.annotation == list: param_type = "array"
        elif param.annotation == dict: param_type = "object"
        
        properties[name] = {
            "type": param_type,
            "description": f"参数 {name}" # 简易处理：如有需要，可通过正则从 doc 中提取更详细的参数说明
        }
        
        # 如果没有默认值，则为必填项
        if param.default == inspect.Parameter.empty:
            required.append(name)
            
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip().replace("\n", " "), # 将多行注释压缩
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    
    # 自动注册
    TOOL_MAP[func.__name__] = func
    TOOLS_SCHEMA.append(schema)
    return func

# --- 具体的执行代码 ---

@register_tool
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

@register_tool
def calculate(expression: str) -> str:
    """计算数学表达式的值。适用于加减乘除等基础数学运算。传入示例：'12 * (3 + 5)'"""
    try:
        if not re.match(r'^[0-9\+\-\*\/\(\)\.\s]+$', expression):
            return "错误：表达式包含非法字符，出于安全考虑已拦截。"
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"数学计算出错: {str(e)}"

# 不再需要手写 TOOLS_SCHEMA，全部由 @register_tool 自动搞定！