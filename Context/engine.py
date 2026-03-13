import json
import traceback # 用于打印详细错误栈
from config import client, Config, logger
from prompts import SYSTEM_PROMPT
from tools import TOOL_MAP, TOOLS_SCHEMA
from database import db
from utils import count_tokens, generate_title, generate_fact_sheet

class ReActAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.system_prompt = SYSTEM_PROMPT
        self.session_title = db.get_session_title(session_id)
        # 初始加载：恢复现场
        self.messages = self._load_context()
        self.current_total_tokens = count_tokens(self.messages) # 初始预估，后续由 API 覆盖
        logger.info(f"成功加载会话 {self.session_title}，当前上下文 Token 数: {self.current_total_tokens}")

    def _load_context(self):
        """从数据库重建上下文，静默加载，仅在恢复历史时打印关键信息"""
        context = [{"role": "system", "content": self.system_prompt}]
        
        # 获取 Summary
        summary_data = db.get_summary(self.session_id)
        last_msg_id = 0
        summary_content = ""
        if summary_data:
            summary_content, last_msg_id = summary_data
            context.append({"role": "system", "content": f"【 Summary 】:\n{summary_content}"})
        
        # 加载 Summary 之后的原始消息（还原最近的对话现场）
        history = db.get_messages_after(self.session_id, last_msg_id)
        context.extend(history)

        # 打印逻辑：仅当存在摘要或历史消息时触发（即非新对话）
        if summary_data or history:
            # 打印摘要
            if summary_content:
                print(f"\n**Summary**:\n{summary_content}")

            # 打印最近对话
            if history:
                print("\n**最近对话历史**:")
                for msg in history:
                    role = msg['role']
                    content = msg.get('content') or ""
                    
                    if role == "user":
                        print(f"👤 User: {content}")
                    
                    elif role == "assistant":
                        # 仅打印 Final Answer 之后的部分
                        if "Final Answer:" in content:
                            final_ans = content.split("Final Answer:")[-1].strip()
                            print(f"🤖 Assistant: {final_ans}")
                        # 如果没有 Final Answer 且没有工具调用，视为普通回答打印
                        elif not msg.get("tool_calls") and content.strip():
                            print(f"🤖 Assistant: {content}")
                    
                    # role == "tool" 的消息将被完全跳过，不打印给用户
            
            print("\n" + "="*50 + "\n")

        return context

    def _safe_json_parse(self, args_str):
        """增强版 JSON 解析：处理模型可能输出的 Markdown 代码块或非法字符"""
        try:
            if isinstance(args_str, dict): return args_str
            # 去除可能存在的 Markdown 标签
            clean_str = args_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str.split("```json")[1].split("```")[0].strip()
            elif clean_str.startswith("```"):
                clean_str = clean_str.split("```")[1].split("```")[0].strip()
            return json.loads(clean_str)
        
        except Exception as e:
            logger.error(f"JSON 解析失败: {args_str} | 错误: {e}")
            return None
        
    def _save_and_append(self, role, content=None, tool_calls=None, tool_call_id=None):
        """统一管理，存入数据库并加入当前内存上下文"""
        msg = {"role": role}
        if content: msg["content"] = content
        if tool_calls: msg["tool_calls"] = tool_calls
        if tool_call_id: msg["tool_call_id"] = tool_call_id
        
        # 持久化到数据库，将 Token 保存设为 0，因为我们将基于整体上下文做拦截，不再强依赖单条记录
        db.save_message(
            self.session_id, role, content, tool_calls, tool_call_id, tokens=0
        )
        # 加入当前对话上下文
        self.messages.append(msg)

    def _check_and_summarize(self):
        """依靠最新的精确 Token 计算进行拦截判断"""
        if self.current_total_tokens > Config.TOKEN_SOFT_LIMIT:
            logger.info(f"\n[系统] Token 数 ({self.current_total_tokens}) 超过软上限，后台开始压缩记忆...")
            
            # 获取当前已有的摘要
            old_summary_data = db.get_summary(self.session_id)
            old_content = old_summary_data[0] if old_summary_data else ""
            
            # 获取自上次摘要后的所有消息进行压缩
            last_id = old_summary_data[1] if old_summary_data else 0
            new_msgs = db.get_messages_after(self.session_id, last_id)
            
            if new_msgs:
                # 调用 LLM 生成新的 Summary 事实清单
                new_fact_sheet = generate_fact_sheet(old_content, new_msgs)
                
                # 找到当前消息表中最后一条消息的 ID
                # 简单处理：取 messages 表中该 session 的最大 ID
                cursor = db.conn.cursor()
                cursor.execute("SELECT MAX(id) FROM messages WHERE session_id = ?", (self.session_id,))
                max_id = cursor.fetchone()[0]
                
                # 更新摘要表
                db.update_summary(self.session_id, new_fact_sheet, max_id)
                logger.info("事实清单已更新，上下文已成功压缩。")
                
                # 重载内存中的上下文，释放 Token 空间
                self.messages = self._load_context()
                self.current_total_tokens = count_tokens(self.messages)

    def run(self, user_query: str, max_turns: int = 5):
        """流式生成器：通过 yield 逐步返回生成的文字和状态"""

        # 处理用户输入
        # 如果是新会话的第一条，尝试生成标题
        current_history = db.get_messages_after(self.session_id)
        if len(current_history) == 0:
            new_title = generate_title(user_query)
            db.update_session_title(self.session_id, new_title)
            logger.info(f"已自动生成会话标题: {new_title}")

            # 更新对话标题
            self.session_title = new_title

        # 将当前用户输入存入数据库并加入当前上下文
        self._save_and_append("user", content=user_query)
        
        for turn in range(max_turns):
            logger.info(f"开始第 {turn + 1} 轮思考...")
            
            try:
                # 开启 stream 模式，并请求携带 usage 信息
                response = client.chat.completions.create(
                    model=Config.MODEL,
                    messages=self.messages,
                    tools=TOOLS_SCHEMA if TOOLS_SCHEMA else None,
                    tool_choice="auto",
                    temperature=0.2, # 流式推理时略微加一点温度让思维更流畅
                    stream=True,
                    stream_options={"include_usage": True} # 让最后一个 chunk 返回精确的 Token 消耗
                )
            except Exception as e:
                yield f"\n[系统错误] API 调用失败: {e}"
                return
            
            # --- 用于拼接流式碎片的容器 ---
            content_buffer = ""
            tool_calls_dict = {}  # 结构：{index: {"id":..., "function": {"name":..., "arguments":...}}}

            for chunk in response:
                # 处理最后一个包含 usage 的纯数据 chunk
                if not chunk.choices and chunk.usage:
                    self.current_total_tokens = chunk.usage.total_tokens
                    continue

                delta = chunk.choices[0].delta

                # 1. 如果是文本流（模型的思考或回答）
                if delta.content:
                    content_buffer += delta.content
                    yield delta.content  # 实时吐出给前端界面

                # 2. 如果是工具调用流（需要碎片拼接）
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name or "", "arguments": ""}
                            }
                        else:
                            if tc.function.name:
                                tool_calls_dict[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments
            # --- 流读取完毕，整理本轮回合数据 ---
            tool_calls_list = list(tool_calls_dict.values()) if tool_calls_dict else None

            # 保存到历史记录中
            self._save_and_append(
                "assistant", 
                content=content_buffer if content_buffer else None, 
                tool_calls=tool_calls_list
            )

            # 如果没有触发工具调用，说明大模型给出了最终回答，跳出循环
            if not tool_calls_list:
                self._check_and_summarize()
                return
            
            # 如果触发了工具调用，开始执行
            for tc in tool_calls_list:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                func_args = self._safe_json_parse(raw_args)
                tool_call_id = tc["id"] # 每个调用都有唯一 ID

                # 提示用户正在使用什么工具
                yield f"\n\n[⚙️ 正在调用工具: {func_name} ...]\n"
                    
                if func_args is None:
                    obs = "错误：工具参数 JSON 格式非法。"
                # 工具执行逻辑
                elif func_name not in TOOL_MAP:
                    obs = f"错误：工具 {func_name} 不存在。"
                else:                        
                    logger.info(f"执行工具: {func_name} | 参数: {func_args}")
                    try:
                        obs = TOOL_MAP[func_name](**func_args)
                    except Exception as e:
                        # 将错误回传给模型，让它尝试修复
                        obs = f"工具执行出错: {str(e)}"
                        logger.error(f"工具 {func_name} 崩溃: {traceback.format_exc()}")
                                            
                    # 把结果喂回给大模型上下文
                    # 必须包含 tool_call_id，角色必须是 "tool"
                    self._save_and_append("tool", content=str(obs), tool_call_id=tool_call_id)
                
                # 工具执行完毕，不要退出，直接 continue 进入下一轮让大模型总结
            continue

        yield "\n\n[系统] 思考达到最大上限，未能完全解决问题。"
        self._check_and_summarize()
    
    