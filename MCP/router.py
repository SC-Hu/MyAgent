import json
from config import client, Config, logger
from tools import TOOLKIT_METADATA

async def route_intent(user_query: str) -> list:
    """
    独立出的技能路由层 (Tool Router 网关)
    根据用户意图，动态决定激活哪些领域的技能包。
    """
    # 构造给 Router 看的菜单
    menu = "\n".join([f"- {name}: {meta}" for name, meta in TOOLKIT_METADATA.items()])
    
    router_prompt = f"""
    你是一个高层意图分配网关。请根据用户的问题，从以下【能力领域】中挑选出解决问题所必需的工具箱名。
    请严格返回 JSON 格式，必须包含 "json" 字样。

    【能力领域菜单】：
    {menu}
    
    【决策准则】：
    1. 如果用户要求写代码、读写文件、操作 Word/Txt，请选择 "office"。
    2. 如果涉及游戏业务逻辑、Unity 报错、数值计算，请选择 "gamedev"。
    3. 如果涉及运行脚本、安装软件、执行终端 Bash 指令，请选择 "system"。
    4. 如果只是普通问答或搜索，返回空列表 []。
    
    【示例 (Few-Shot)】：
    {{ "active_toolkits": ["工具箱名1", "工具箱名2"] }}
    {{"active_skills": ["office","system"]}}
    """

    try:
        # 换成成本低的小模型，这仅仅是个分类任务！
        resp = await client.chat.completions.create(
            model=Config.MODEL, 
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": user_query}
            ],
            # 核心优化，开启严格的 JSON Schema 物理级约束
            # 很多模型不支持以下模式！ ---
            # response_format={"type": "json_schema",
            #     "json_schema": {
            #         "name": "SkillRouterResponse",
            #         "strict": True,  # 开启严格模式，拒绝任何额外字段
            #         "schema": {
            #             "type": "object",
            #             "properties": {
            #                 "active_skills": {
            #                     "type": "array",
            #                     "description": "根据用户意图需要激活的技能包列表",
            #                     "items": {
            #                         "type": "string",
            #                         # 强行限制数组里的元素只能是注册过的 valid_skills！
            #                         "enum": valid_skills 
            #                     }
            #                 }
            #             },
            #             "required":["active_skills"], # 声明该字段必须存在
            #             "additionalProperties": False  # 绝对禁止模型自己发明其他的 Key
            #         }
            #     }
            # },
            response_format={"type": "json_object"},
            temperature=0  # 分类任务必须是 0 逻辑，拒绝任何发散
        )

        # 获取大模型的原始回复
        raw_content = resp.choices[0].message.content
        # 直接把大模型的原始字符串打印出来
        print(f"\n[Router] 模型原始输出: {raw_content}")
        
        # json schema：此时拿到的内容，100% 绝对是合法的 JSON，且绝对不包含不存在的技能名
        # json object：可能出错，使用 Few-Shot
        result = json.loads(resp.choices[0].message.content)
        # 兼容大模型可能输出 active_skills 或 active_toolkits
        skills = result.get("active_toolkits") or result.get("active_skills") or []
        return skills
    
    except Exception as e:
        logger.error(f"Router 网关异常: {e}")
        # 如果路由网关崩溃，采取降级策略：默认不加载附加技能包，只用 base 保底
        return[]