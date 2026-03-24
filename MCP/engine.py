import json
import traceback
import inspect # 用于检查工具是同步还是异步
import asyncio
from langfuse import observe
from config import client, Config, logger
from prompts import SYSTEM_PROMPT, REFLECTION_PROMPT
from database import db
from utils import count_tokens, generate_title, generate_fact_sheet
from tools import TOOLKIT_REGISTRY
from router import route_intent  # 引入路由网关
from memory_manager import long_term_memory


class ReActAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.system_prompt = SYSTEM_PROMPT
        self.session_title = db.get_session_title(session_id)
        # 初始加载大模型的上下文（静默进行）
        self.messages = self._load_context()
        self.current_total_tokens = count_tokens(self.messages) # 初始预估，后续由 API 覆盖


    def _load_context(self):
        """从数据库重建上下文，纯静默加载，专门提供给大模型看"""
        context = [{"role": "system", "content": self.system_prompt}]
        
        # 获取 Summary
        summary_data = db.get_summary(self.session_id)
        last_msg_id = 0
        if summary_data:
            summary_content, last_msg_id = summary_data
            context.append({"role": "system", "content": f"【 Summary 】:\n{summary_content}"})
        
        # 只加载摘要之后的消息，实现滑动窗口压缩
        history = db.get_messages_after(self.session_id, last_msg_id)
        context.extend(history)
        return context


    def show_chat_history(self):
        """给用户看的全量历史记录"""
        history = db.get_full_chat_history(self.session_id)
        if not history: return
        
        for role, content, tool_calls_json in history:
            if role == "user":
                print(f"👤 User: {content}")
                
            elif role == "assistant":
                # 打印思考流 
                if content:
                    # 清理可能存在的颜色转义符，让历史记录看起来干净点
                    clean_thought = content.replace("\n\033[90m[🧠 思考流] ", "").replace("\033[0m", "").strip()
                    # 用一个稍微淡一点的颜色或者前缀标识这是思考过程
                    print(f"   (🧠 思考: {clean_thought[:100]}...)") # 历史记录里可以只看个开头
                # 提取并打印最终答案 (Final Answer)
                if tool_calls_json:
                    try:
                        tool_calls = json.loads(tool_calls_json)
                        for tc in tool_calls:
                            if tc['function']['name'] == "submit_final_answer":
                                # 解析工具参数中的 answer
                                args = json.loads(tc['function']['arguments'])
                                final_ans = args.get('answer', '[未提取到答案内容]')
                                print(f"🤖 Assistant: {final_ans}")
                    except Exception as e:
                        # 如果解析失败，可能是脏数据，跳过
                        pass
        

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
        # 防止空字符串被丢弃，引发 API 格式错误
        if content is not None: msg["content"] = content
        if tool_calls: msg["tool_calls"] = tool_calls
        if tool_call_id: msg["tool_call_id"] = tool_call_id
        
        # 持久化到数据库，将 Token 保存设为 0，因为我们将基于整体上下文做拦截，不再强依赖单条记录
        db.save_message(
            self.session_id, role, content, tool_calls, tool_call_id, tokens=0
        )
        # 加入当前对话上下文
        self.messages.append(msg)


    # --- 重构：统一记忆生命周期中枢 ---
    async def sync_memories(self, force=False):
        """
        统管短期上下文压缩与长期记忆沉淀。
        触发条件：Token 超过软上限，或者强制触发 (force=True，如退出程序时)。
        """
        # 如果既没有超标，也不是强制退出，则什么都不做
        if self.current_total_tokens <= Config.TOKEN_SOFT_LIMIT and not force:
            return

        trigger_reason = "退出整理" if force else f"Token 超标({self.current_total_tokens})"
        logger.info(f"\n[系统] {trigger_reason}，记忆中枢开始同步数据...")

        # 获取尚未被处理的新对话记录
        old_summary_data = db.get_summary(self.session_id)
        new_msgs = db.get_messages_after(self.session_id, old_summary_data[1] if old_summary_data else 0)

        if not new_msgs:
            return

        if new_msgs:
            # --- 存入 SQLite 摘要表 (保持连贯性) ---
            new_fact_sheet = await generate_fact_sheet(old_summary_data[0] if old_summary_data else "", new_msgs)
            max_id = db.save_message(self.session_id, "system", content="摘要同步") # 临时占位获取 ID 或直接用 SQL max
            db.update_summary(self.session_id, new_fact_sheet, max_id)
            
            # --- 存入 ChromaDB (沉淀为跨会话的长期记忆) ---
            messages_str = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in new_msgs])
            long_term_memory.save_facts(messages_str) # 自动完成向量化与存储
            
            self.messages = self._load_context()
            self.current_total_tokens = count_tokens(self.messages)


    @observe(as_type="generation") # 追踪反思过程
    async def _run_reflection(self, user_query: str, draft_answer: str, retrieved_memories: str = "") -> tuple[bool, str]:
        """自我反思机制：使用 JSON Mode 确保返回格式"""
        # 组装带有 RAG 记忆的审核背景
        memory_context = f"【系统附加的长期记忆】:\n{retrieved_memories}\n\n" if retrieved_memories else ""

        prompt = f"{memory_context}【用户原始问题】: {user_query}\n\n【Agent草稿回答】: {draft_answer}"
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
    
    @observe() # 自动追踪整个 run 函数的执行链路
    async def run(self, user_query: str, max_turns: int = 10): # 增加轮数以支持反思循环
        """流式生成器：通过 yield 逐步返回生成的文字和状态"""

        # 处理用户输入
        # 如果是新会话的第一条，尝试生成标题
        current_history = db.get_messages_after(self.session_id)
        if len(current_history) == 0:
            new_title = await generate_title(user_query)
            db.update_session_title(self.session_id, new_title)

            # 更新对话标题
            self.session_title = new_title

        # 把原始输入存入 SQLite (总账本，保证信息完整性)
        db.save_message(self.session_id, "user", content=user_query, tokens=0)

        yield "\n\033[36m[🧠 记忆神经] 正在潜意识中检索相关记忆...\033[0m"
        # 根据用户的提问，去向量库里搜相关的记忆
        retrieved_memories = await asyncio.to_thread(long_term_memory.retrieve, user_query)
        
        # 构建专属于本次 API 请求的临时会话列表 (浅拷贝)
        request_messages = list(self.messages)

        # 组装最终喂给大模型的 User Prompt（隐式上下文注入）
        if retrieved_memories:
            yield f"\n\033[36m[💡 记忆唤醒] 找到相关过往背景！\033[0m"
            request_messages.append({
                "role": "system",
                "content": f"【系统附加长期记忆(供参考)】:\n{retrieved_memories}"
            })

        request_messages.append({"role": "user", "content": user_query})
        self.messages.append({"role": "user", "content": user_query})
        
        # --- 核心修改：三级工具加载流 ---
        # 1. Router 选赛道 (Toolkit Names)
        yield "\n\033[36m[🧭 系统路由] 正在匹配能力领域...\033[0m"
        
        # 直接调用从 router.py 引入的函数
        active_domains = await route_intent(user_query)

        # 看 Router 放行了什么技能
        print(f"\n[Router] 判决激活的技能包: {active_domains}")
        
        # 2. Tool RAG 选重点并激活箱子
        current_tool_map = {}
        current_tools_schema = []

        # 基础包永远加载
        current_tool_map.update(TOOLKIT_REGISTRY["base"]["tools"])
        current_tools_schema.extend(TOOLKIT_REGISTRY["base"]["schemas"])

        if active_domains:
            yield "\n\033[36m[🔍 工具检索] 正在从领域内筛选最匹配的工具箱...\033[0m"
            # 这里的 RAG 检索会返回相关的 Toolkit 名字（整箱加载策略）
            matched_tk_names = long_term_memory.search_toolkits(user_query, active_domains)
            
            yield f"\n\033[36m[📦 挂载完成] 已激活工具箱: {matched_tk_names}\033[0m"
            
            for tk_name in matched_tk_names:
                if tk_name in TOOLKIT_REGISTRY:
                    current_tool_map.update(TOOLKIT_REGISTRY[tk_name]["tools"])
                    current_tools_schema.extend(TOOLKIT_REGISTRY[tk_name]["schemas"])


        for turn in range(max_turns):
            logger.info(f"\n[系统] 开始第 {turn + 1} 轮思考...")
            try:
                # 开启 stream 模式，并请求携带 usage 信息
                response = await client.chat.completions.create(
                    model=Config.MODEL,
                    messages=request_messages, # 使用带有临时记忆和最新消息的请求列表
                    tools=current_tools_schema, # 不再传全局 SCHEMA，而是传刚才动态拼装的
                    tool_choice="auto",
                    temperature=0.3, # 促进思考
                    stream=True,
                    stream_options={"include_usage": True} # 让最后一个 chunk 返回精确的 Token 消耗
                )
            except Exception as e:
                yield f"\n[系统错误] API 调用失败: {e}"
                return

            # 用于拼接流式碎片的容器
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

            # 1. 保存助手回复到内存和数据库
            self._save_and_append(
                "assistant", 
                content=content_buffer if content_buffer else None, 
                tool_calls=tool_calls_list
            )
            request_messages.append(self.messages[-1])

            # 2. 如果没有工具调用，且有文本，说明回答完毕
            if not tool_calls_list and content_buffer:
                # 只有文本，不需要 tool 回复，直接走柔性兜底逻辑
                pass 
            
            # 多工具并行执行
            if tool_calls_list:
                # 任务分类，将“常规工具”和“交卷工具”分开
                normal_tool_calls = []
                # 把单一变量改成列表，防止相互覆盖
                submit_calls = []
                for tc in tool_calls_list:
                    if tc["function"]["name"].endswith("submit_final_answer"):
                        submit_calls.append(tc)
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
                        tool_call_id = tc["id"]
                        
                        if func_args is None:
                            return tool_call_id, f"错误：JSON 格式非法: {raw_args}"
                        
                        # --- 从嵌套字典中提取函数和审批元数据 ---
                        tool_entry = current_tool_map.get(func_name)
                        if not tool_entry: return tool_call_id, f"错误：工具 {func_name} 未在当前工作区注册。"

                        target_func = tool_entry["func"]
                        requires_approval = tool_entry["requires_approval"]

                        # --- 核心修改，人工审批 (HITL) 拦截器 ---
                        if requires_approval:
                            print(f"\n\033[33m[⚠️ 审批申请] Agent 计划执行高危操作: {func_name}\033[0m")
                            print(f"\n参数详情: {json.dumps(func_args, indent=2, ensure_ascii=False)}")
                            
                            # 阻塞等待用户 y/n (由于是多工具并发，一次只能审批一个)
                            choice = await asyncio.to_thread(input, "\n👉 是否批准执行？(y/n): ")
                            if choice.lower() != 'y':
                                print(f"\n\033[31m[系统] 操作 {func_name} 已被拒绝。\033[0m")
                                return tool_call_id, "用户拒绝了此操作执行。请告知用户你理解了并寻找更安全的方法。"
                            
                        print(f"\n[⚙️ 执行中: {func_name} ...]")

                        try:
                            # 反射注入 Agent 上下文
                            sig = inspect.signature(target_func)
                            if "agent_context" in sig.parameters:
                                func_args["agent_context"] = self # 把 Agent 自己传进去

                            if inspect.iscoroutinefunction(target_func):
                                # 如果是 async def 工具，等待
                                res = await target_func(**func_args)
                            else:
                                # 如果是普通的 def 工具（如繁重的计算或读写文件），
                                # 把它扔进底层的线程池执行，绝对不阻塞 Async 主线程
                                res = await asyncio.to_thread(target_func, **func_args)
                            return tool_call_id, str(res)
                        except Exception as e:
                            error_detail = traceback.format_exc()
                            return tool_call_id, f"工具执行异常:\n{error_detail}\n请修正后重新尝试！"
                                
                    # 使用 asyncio.gather 瞬间同时启动所有常规工具
                    tasks = [execute_single_tool(tc) for tc in normal_tool_calls]
                    results = await asyncio.gather(*tasks)

                    # 将所有并发执行的结果批量存入数据库和上下文
                    for tool_call_id, obs in results:
                        self._save_and_append("tool", content=obs, tool_call_id=tool_call_id)
                        request_messages.append(self.messages[-1])
                    
                    # 如果这轮没有交卷，则继续下一轮思考
                    if not submit_calls:
                        continue

                # 独立处理交卷工具 (Reflection 拦截)
                if submit_calls:
                    for submit_call in submit_calls:
                        func_name = submit_call["function"]["name"]
                        raw_args = submit_call["function"]["arguments"]
                        func_args = self._safe_json_parse(raw_args)
                        tool_call_id = submit_call["id"]
                        
                        draft_answer = func_args.get("answer", "未提取到答案。") if func_args else "解析失败"
                        yield f"\n\n[🕵️ 系统审核拦截 (Self-Reflection)...]\n"
                            
                        # 触发大模型审视草稿
                        is_pass, feedback = await self._run_reflection(user_query, draft_answer, retrieved_memories)
                            
                        if is_pass:
                            yield f"✅ 审核通过！\n\n🎯 最终回答:\n"
                            # 打字机特效输出最终答案
                            for char in draft_answer:
                                yield char
                                await asyncio.sleep(0.01)

                            self._save_and_append("tool", content="审核通过。任务结束。", tool_call_id=tool_call_id)
                            # --- 同步更新 request_messages ---
                            request_messages.append(self.messages[-1]) 
                            await self.sync_memories(force=False) # 正常检查是否超 Token
                            return # 真正结束并退出
                        else:
                            yield f"❌ 审核被驳回：{feedback}\n[🔄 触发自愈...]\n"
                            obs = f"回答被系统 Reviewer 驳回！必须修正，驳回理由：\n{feedback}"
                            self._save_and_append("tool", content=obs, tool_call_id=tool_call_id)
                            # --- 必须把这条驳回记录加入到临时账本里，API 才能正常往下走 ---
                            request_messages.append(self.messages[-1])
                    
                    # 只有当所有的 submit_call 都被驳回时，才会走到这里，继续进入下一轮循环重试
                    continue 
            
            else:
                # --- 柔性兜底 (刺头模型非要直接说话) ---
                if not content_buffer.strip():
                    continue # 连字都没打，直接跳过
                    
                yield f"\n\n[🕵️ 强制系统审核拦截 (Fallback Reflection)...]\n"
                draft_answer = content_buffer
                
                is_pass, feedback = await self._run_reflection(user_query, draft_answer, retrieved_memories)
                    
                if is_pass:
                    yield f"✅ 审核通过！(触发柔性兜底)\n\n🎯 最终回答:\n"
                    # 打字机特效输出
                    for char in draft_answer:
                        yield char
                        await asyncio.sleep(0.01)
                        
                    await self.sync_memories(force=False)
                    return 
                else:
                    yield f"❌ 审核被驳回：{feedback}\n[🔄 触发自愈...]\n"
                    obs = f"你刚才直接输出的回答被审核员驳回！\n驳回理由：{feedback}\n请修正后重新作答。你可以直接回答，也可以使用工具。"
                    self._save_and_append("system", content=obs)
                    request_messages.append(self.messages[-1])
                    continue


        yield "\n\n[系统] 思考达到最大上限，未能完全解决问题。"
        await self.sync_memories(force=False)
    
    