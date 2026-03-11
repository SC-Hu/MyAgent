import re
from config import client, Config, logger
from prompts import SYSTEM_PROMPT
from tools import TOOL_MAP

class ReActAgent:
    def __init__(self):
        # 1. 动态生成工具描述，填入提示词模板
        tool_descriptions = "\n".join([f"- {name}: {func.__doc__}" for name, func in TOOL_MAP.items()])
        self.system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        
        # 2. 初始化对话记忆
        self.messages = [{"role": "system", "content": self.system_prompt}]
        
        # 3. 准备正则表达式（用来从 LLM 的回复中“抠出”工具名和参数）
        self.action_re = re.compile(r'^Action:\s*(.*?)$', re.MULTILINE)
        self.action_input_re = re.compile(r'^Action Input:\s*(.*?)$', re.MULTILINE)

    def run(self, user_query: str, max_turns: int = 5):
        self.messages.append({"role": "user", "content": user_query})
        
        for turn in range(max_turns):
            logger.info(f"开始第 {turn + 1} 轮思考...")
            
            # 1. 让模型思考，temp设为0，让 Agent 的逻辑高度稳定
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=self.messages,
                temperature=0
            )
            
            content = response.choices[0].message.content
            logger.info(f"模型原始输出: \n{content}")

            self.messages.append({"role": "assistant", "content": content})

            # 2. 检查是否得到最终答案
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[-1].strip()
                logger.info("任务完成，准备输出结果。")
                return final_answer

            # 3. 解析并执行工具
            action = self.action_re.search(content)
            action_input = self.action_input_re.search(content)

            if action and action_input:
                tool_name = action.group(1).strip()
                tool_args = action_input.group(1).strip()
                
                # 4. 执行工具并获取“观察结果”
                if tool_name in TOOL_MAP:
                    logger.info(f"正在调用工具: {tool_name}，参数: {tool_args}")
                    observation = TOOL_MAP[tool_name](tool_args)
                    logger.info(f"工具返回结果: {observation}")

                    # 5. 关键：把结果喂回给模型
                    self.messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    logger.warning(f"模型尝试调用不存在的工具: {tool_name}")
                    self.messages.append({"role": "user", "content": f"Observation: Tool {tool_name} not found."})

            else:
                # 兜底逻辑：如果模型没按格式说话
                logger.info("模型未触发工具，请求补充格式。")
                self.messages.append({"role": "user", "content": "请继续按照格式进行，或给出 Final Answer。"})
        return "思考达到上限，未能解决问题。"