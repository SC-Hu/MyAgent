import json
import traceback
import inspect # 新增 - 用于检查工具是同步还是异步
import asyncio
from langfuse import observe
from config import client, Config, logger
from prompts import SYSTEM_PROMPT, REFLECTION_PROMPT
from database import db
from utils import count_tokens, generate_title, generate_fact_sheet
from tools import SKILL_REGISTRY
from router import route_intent  # 引入路由网关


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

    async def _check_and_summarize(self):
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
                new_fact_sheet = await generate_fact_sheet(old_content, new_msgs)
                
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

    @observe(as_type="generation") # 追踪反思过程
    async def _run_reflection(self, user_query: str, draft_answer: str) -> tuple[bool, str]:
        """自我反思机制：使用 JSON Mode 确保返回格式"""
        prompt = f"【用户原始问题】: {user_query}\n\n【Agent草稿回答】: {draft_answer}"
        try:
            resp = await client.chat.completions.create(
                model=Config.MODEL, # 此处工业界常替换为稍便宜但逻辑好的模型以省钱
                messages=[
                    {"role": "system", "content": REFLECTION_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # 强约束输出 JSON
                temperature=0.1
            )
            result = json.loads(resp.choices[0].message.content)
            return result.get("pass", False), result.get("feedback", "解析失败")
        except Exception as e:
            logger.error(f"反思模块异常: {e}")
            return True, "反思模块异常，默认放行"
    
    @observe() # 新增 - 自动追踪整个 run 函数的执行链路
    async def run(self, user_query: str, max_turns: int = 10): # 增加轮数以支持反思循环
        """流式生成器：通过 yield 逐步返回生成的文字和状态"""

        # 处理用户输入
        # 如果是新会话的第一条，尝试生成标题
        current_history = db.get_messages_after(self.session_id)
        if len(current_history) == 0:
            new_title = await generate_title(user_query)
            db.update_session_title(self.session_id, new_title)
            logger.info(f"已自动生成会话标题: {new_title}")

            # 更新对话标题
            self.session_title = new_title

        # 将当前用户输入存入数据库并加入当前上下文
        self._save_and_append("user", content=user_query)

        
        # --- 调用 Router 网关 ---
        yield "正在分析任务意图，装载技能包..."
        
        # 直接调用从 router.py 引入的函数
        active_skills = await route_intent(user_query)
        
        # 拼接基础技能
        skills_to_load = ["base"] + active_skills
        yield f"技能装载完成，当前激活模块: {skills_to_load}"

        current_tool_map = {}
        current_tools_schema = []
        for skill in set(skills_to_load):
            if skill in SKILL_REGISTRY:
                current_tool_map.update(SKILL_REGISTRY[skill]["tools"])
                current_tools_schema.extend(SKILL_REGISTRY[skill]["schemas"])


        for turn in range(max_turns):
            logger.info(f"开始第 {turn + 1} 轮思考...")
            
            try:
                # 开启 stream 模式，并请求携带 usage 信息
                response = await client.chat.completions.create(
                    model=Config.MODEL,
                    messages=self.messages,
                    tools=current_tools_schema, # --- 不再传全局 SCHEMA，而是传刚才动态拼装的 ---
                    tool_choice="auto",
                    temperature=0.3, # 促进思考
                    stream=True,
                    stream_options={"include_usage": True} # 让最后一个 chunk 返回精确的 Token 消耗
                )
            except Exception as e:
                yield f"\n[系统错误] API 调用失败: {e}"
                return

            # --- 用于拼接流式碎片的容器 ---
            content_buffer = ""
            tool_calls_dict = {}  # 结构：{index: {"id":..., "function": {"name":..., "arguments":...}}}
            content_started = False # 终端UI渲染开关

            async for chunk in response:
                # 处理最后一个包含 usage 的纯数据 chunk
                if not chunk.choices and chunk.usage:
                    self.current_total_tokens = chunk.usage.total_tokens
                    continue

                delta = chunk.choices[0].delta

                # 遇到文本，打印思维链（CoT）
                if delta.content:
                    if not content_started:
                        yield "\n\033[90m[🧠 思考流] " # 使用 ANSI 灰色打印思考过程
                        content_started = True
                    content_buffer += delta.content
                    yield delta.content  # 实时吐出给前端界面

                # 遇到工具调用
                if delta.tool_calls:
                    if content_started:
                        yield "\033[0m" # 恢复默认终端颜色
                        content_started = False

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
            
            # 恢复颜色，如果只输出了文本，没有调工具，也要保证在最后把颜色恢复正常，不影响用户后续的打字
            if content_started: yield "\033[0m"

            # 流读取完毕，整理本轮回合数据
            tool_calls_list = list(tool_calls_dict.values()) if tool_calls_dict else None

            # 保存到历史记录中
            self._save_and_append(
                "assistant", 
                content=content_buffer if content_buffer else None, 
                tool_calls=tool_calls_list
            )

            # 异常情况：模型既没思考也没调用工具
            if not tool_calls_list and not content_buffer:
                yield "\n[系统] 模型返回为空，结束思考。"
                break
            
            # --- 核心改进 多工具并行执行 ---
            if tool_calls_list:
                # 任务分类，将“常规工具”和“交卷工具”分开
                normal_tool_calls = []
                submit_call = None
                for tc in tool_calls_list:
                    if tc["function"]["name"] == "submit_final_answer":
                        submit_call = tc
                    else:
                        normal_tool_calls.append(tc)

                # 并发执行常规工具
                if normal_tool_calls:
                    tool_names = [tc["function"]["name"] for tc in normal_tool_calls]
                    yield f"\n[⚡ 并发调用 {len(normal_tool_calls)} 个工具: {', '.join(tool_names)} ...]"
                    
                    # 内部异步函数，用于包装单个工具的执行逻辑
                    async def execute_single_tool(tc):
                        func_name = tc["function"]["name"]
                        raw_args = tc["function"]["arguments"]
                        func_args = self._safe_json_parse(raw_args)
                        tool_call_id = tc["id"] # 每个调用都有唯一 ID
                    
                        yield f"\n\n[⚙️ 正在调用工具: {func_name} ...]\n"
                        
                        if func_args is None:
                            return tool_call_id, f"错误：工具参数 JSON 格式非法: {raw_args}。请修正你的输出格式。"
                        # 工具执行逻辑
                        elif func_name not in current_tool_map:
                            return tool_call_id, f"错误：工具 {func_name} 不存在。"
                        else:                        
                            logger.info(f"执行工具: {func_name} | 参数: {func_args}")
                            try:
                                # 智能工具执行，兼容同步和异步工具
                                target_func = current_tool_map[func_name]

                                # --- 核心改进，反射注入 Agent 上下文 ---
                                sig = inspect.signature(target_func)
                                if "agent_context" in sig.parameters:
                                    func_args["agent_context"] = self # 把 Agent 自己传进去

                                if inspect.iscoroutinefunction(target_func):
                                    # 如果是 async def 工具，原生等待
                                    res = await target_func(**func_args)
                                else:
                                    # 如果是普通的 def 工具（如繁重的计算或读写文件），
                                    # 把它扔进底层的线程池执行，绝对不阻塞 Async 主线程！
                                    res = await asyncio.to_thread(func_name, **func_args)
                                return tool_call_id, str(res)
                            except Exception as e:
                                error_detail = traceback.format_exc()
                                logger.error(f"工具 {func_name} 崩溃: {error_detail}")
                                return tool_call_id, f"工具执行异常:\n{error_detail}\n请修正后重新尝试！"
                                
                    # 使用 asyncio.gather 瞬间同时启动所有常规工具
                    tasks = [execute_single_tool(tc) for tc in normal_tool_calls]
                    results = await asyncio.gather(*tasks)

                    # 将所有并发执行的结果批量存入数据库和上下文
                    for tool_call_id, obs in results:
                        self._save_and_append("tool", content=obs, tool_call_id=tool_call_id)
                    
                    # 如果这轮没有交卷，则继续下一轮思考
                    if not submit_call:
                        continue

                # 独立处理交卷工具 (Reflection 拦截)
                if submit_call:
                    func_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    func_args = self._safe_json_parse(raw_args)
                    tool_call_id = tc["id"]
                    
                    draft_answer = func_args.get("answer", "未提取到答案。") if func_args else "解析失败"
                    yield f"\n\n[🕵️ 系统审核拦截 (Self-Reflection)...]\n"
                        
                        # 触发大模型审视草稿
                    is_pass, feedback = await self._run_reflection(user_query, draft_answer)
                        
                    if is_pass:
                        yield f"✅ 审核通过！\n\n🎯 最终回答:\n{draft_answer}"
                        self._save_and_append("tool", content="审核通过。任务结束。", tool_call_id=tool_call_id)
                        await self._check_and_summarize()
                        return # 真正结束并退出
                    else:
                        yield f"❌ 审核被驳回：{feedback}\n[🔄 触发自愈纠错机制...]\n"
                        obs = f"你的回答被系统 Reviewer 驳回！必须修正，驳回理由：\n{feedback}"
                        self._save_and_append("tool", content=obs, tool_call_id=tool_call_id)
                        continue # 直接进入下一轮重试
            
            else:
                # 防御机制：如果模型忘记调用 submit_final_answer 直接输出了文本
                yield f"\n[⚠️ 警告：检测到违规输出。已强制要求模型使用标准工具提交答案。]"
                self._save_and_append("user", content="系统提示：绝不要直接在文本中回答用户！请将你的结论通过调用 `submit_final_answer` 工具进行提交。")
                continue


        yield "\n\n[系统] 思考达到最大上限，未能完全解决问题。"
        await self._check_and_summarize()
    
    