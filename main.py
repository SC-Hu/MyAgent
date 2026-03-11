from engine import ReActAgent

def main():
    # 1. 实例化 Agent
    # 这一步会自动调用 engine.py 里的 __init__
    # 完成 Prompt 加载、工具注入、API 客户端初始化
    agent = ReActAgent()
    
    print("🚀 Agent 工程师系统已就绪...")
    print("--- 提示：输入 'exit' 或 'quit' 退出程序 ---")

    # 2. 开启交互循环 (让你可以连续问问题)
    while True:
        user_input = input("\n👤 您想问什么: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("👋 再见！")
            break
            
        if not user_input.strip():
            print("输入不能为空，请重新输入。")
            continue

        try:
            # 3. 核心驱动
            # 调用 engine 里的 run 方法，开始 ReAct 思考循环
            result = agent.run(user_input)
            
            print("\n✅ 最终回答:")
            print(result)
            
        except Exception as e:
            print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    main()