"""
精确计算 Token、标题生成、Summary 生成
"""

import tiktoken
from config import Config, client

def count_tokens(messages):
    """
    精确计算 message 列表消耗的 Token 数量
    """
    try:
        encoding = tiktoken.get_encoding(Config.TOKEN_ENCODING)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # 每条消息的基础消耗
        num_tokens += 4 
        for key, value in message.items():
            if value:
                # 如果是 tool_calls，需要特殊处理（简单转字符串计算，工业级会更复杂点，但误差可接受）
                val_str = str(value)
                num_tokens += len(encoding.encode(val_str))
                if key == "name":  # 如果有名字字段，额外消耗
                    num_tokens += -1
    num_tokens += 2  # 每个回复都以 assistant 开始
    return num_tokens

def generate_title(user_query):
    """根据第一轮对话生成简短标题"""
    prompt = f"根据用户的这个问题，给出一个10字以内的核心主题作为会话标题，不要包含标点。问题：{user_query}"
    try:
        resp = client.chat.completions.create(
            model=Config.SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip().replace("“", "").replace("”", "")
    except:
        return "新会话"
    
def generate_fact_sheet(old_summary, new_messages):
    """生成事实清单式的摘要"""
    messages_str = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in new_messages])
    prompt = f"""请将以下对话中的核心事实、具体数值、技术参数和已确定的结论提取出来。
要求：
1. 保持简洁。
2. 结合旧的背景信息（如果有），更新任务进度。
3. 保留重要的实体名词（如人名、公司名、代码变量名）。

旧背景：{old_summary if old_summary else "无"}
新对话：
{messages_str}
"""
    try:
        resp = client.chat.completions.create(
            model=Config.SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"摘要生成失败: {e}"
