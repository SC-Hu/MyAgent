from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tools import TOOLKIT_REGISTRY, Config
import json

class MCPManager:
    def __init__(self):
        self.sessions = [] # 存储所有的 MCP 客户端连接

    async def connect_to_server(self, category: str, command: str, args: list):
        """
        连接到一个 MCP Server 并将其工具注册进 TOOLKIT_REGISTRY
        category: 注册到哪个技能包 (如 office, system)
        command: 启动命令 (如 npx)
        args: 命令参数
        """
        server_params = StdioServerParameters(command=command, args=args)
        
        # 建立连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. 发现工具
                tools_result = await session.list_tools()
                
                # 2. 遍历并包装工具
                for tool in tools_result.tools:
                    # 我们定义一个“代理函数”，当大模型调用这个工具时，
                    # 实际上是通过 session.call_tool 发送给 Node.js 进程去执行
                    async def mcp_proxy_func(**kwargs):
                        # 这里的 session 必须保持存活
                        res = await session.call_tool(tool.name, arguments=kwargs)
                        return res.content[0].text if res.content else "执行完毕"

                    # 3. 注入 TOOLKIT_REGISTRY
                    # 注意：我们给 MCP 工具加个前缀 mcp_，防止命名冲突
                    mcp_tool_name = f"mcp_{tool.name}"
                    
                    TOOLKIT_REGISTRY[category]["tools"][mcp_tool_name] = {
                        "func": mcp_proxy_func,
                        "requires_approval": True # 外部工具统一要求审批（保险起见）
                    }
                    
                    TOOLKIT_REGISTRY[category]["schemas"].append({
                        "type": "function",
                        "function": {
                            "name": mcp_tool_name,
                            "description": f"[MCP] {tool.description}",
                            "parameters": tool.inputSchema
                        }
                    })