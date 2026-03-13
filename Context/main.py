from engine import ReActAgent
from database import db
from config import Config
from utils import count_tokens


def print_help():
    print("\n" + "="*30)
    print("Agent 指令")
    print("="*30)
    print("/new      - 开启一个全新的会话")
    print("/resume   - 查看并恢复历史会话")
    print("/info     - 查看当前会话状态（Token、ID等）")
    print("/exit     - 退出并保存当前状态")
    print("="*30 + "\n")


def handle_resume():
    """处理恢复历史会话的逻辑"""
    sessions = db.get_recent_sessions(limit=10)
    if not sessions:
        print("暂无历史会话记录。")
        return None

    print("\n最近的历史会话:")
    for idx, (s_id, title, updated_at) in enumerate(sessions):
        print(f"[{idx + 1}] - {title} - ({updated_at})")
    
    choice = input("\n请选择要恢复的序号 (输入 c 取消): ").strip()
    if choice.lower() == 'c':
        return None
    
    try:
        selected_idx = int(choice) - 1
        if 0 <= selected_idx < len(sessions):
            return sessions[selected_idx][0] # 返回 session_id
        else:
            print("无效的序号。")
    except ValueError:
        print("请输入数字。")
    return None


def main():
    # 完成 本地数据库创建、Agent初始化
    # 初始默认开启一个新会话
    current_session_id = db.create_session(title="新会话")
    agent = ReActAgent(current_session_id)

    print("🌟 智能体已启动。输入 / 查看功能。")
    print("--- 提示：输入 'exit' 或 'quit' 退出程序 ---")

    # 开启交互循环
    while True:
        try:
            # 对话框显示当前 Session ID
            user_input = input(f"\n[{agent.session_title}] >>> ").strip()

            if not user_input:
                continue

            # --- 指令拦截器 ---
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()

                if cmd in ["/help", "/"]:
                    print_help()
                    continue

                elif cmd == "/exit":
                    print("正在整理记忆并退出...")
                    agent._check_and_summarize() # 退出前进行一次 Summary 检查
                    print("再见！")
                    break

                elif cmd == "/info":
                    # 从内存和数据库计算当前状态
                    tokens = count_tokens(agent.messages)
                    print(f"\n当前会话信息:")
                    print(f"- Session ID: {agent.session_id}")
                    print(f"- 内存中消息数: {len(agent.messages)}")
                    print(f"- 当前上下文 Token: {tokens} / {Config.TOKEN_SOFT_LIMIT}")
                    continue

                elif cmd == "/new":
                    current_session_id = db.create_session(title="新会话")
                    agent = ReActAgent(current_session_id)
                    print(f"已开启新会话: {current_session_id}")
                    continue

                elif cmd == "/resume":
                    selected_id = handle_resume()
                    if selected_id:
                        agent = ReActAgent(selected_id)
                        print(f"已切换至会话: {selected_id}")
                    continue

                else:
                    print(f"未知指令: {cmd}。输入 / 查看帮助。")
                    continue

            # --- 处理普通对话 (保留原有 exit 逻辑) ---
            if user_input.lower() in ['exit', 'quit']:
                agent._check_and_summarize()
                print("👋 再见！")
                break

            if not user_input.strip():
                print("输入不能为空，请重新输入。")
                continue

            # --- 核心驱动，流式输出 ---
            print("\n🤖 Assistant: ", end="")
            
            # 使用 for 循环逐个接收产出的字符/状态提示
            for chunk in agent.run(user_input):
                if chunk:
                    # flush=True 保证内容立刻被推送到终端显示
                    print(chunk, end="", flush=True)

            print() # 本轮回答完全结束后，换行
            
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()