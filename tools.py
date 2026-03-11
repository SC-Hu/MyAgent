def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression.replace(" ", "")))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """查询天气"""
    mock_data = {"北京": "晴, 25°C", "上海": "阴, 20°C"}
    return mock_data.get(city, f"找不到 {city} 的天气")

# 暴露给 Agent 的工具集
TOOL_MAP = {
    "calculate": calculate,
    "get_weather": get_weather
}