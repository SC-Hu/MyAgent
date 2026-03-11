import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
# 建议安装 pip install python-dotenv 来自动加载 .env 文件

# 这一行会自动寻找当前目录下的 .env 文件并加载
load_dotenv()

# --- 配置 Logging ---
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(levelname)s - %(message)s",
    format="%(message)s",
    handlers=[
        # 记录到文件，供以后查账
        logging.FileHandler("agent_debug.log", encoding="utf-8"), 
        # 同时输出到控制台
        logging.StreamHandler() 
    ]
)
# 屏蔽第三方库的冗余信息
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("AgentEngine")

class Config:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL") 
    MODEL = os.getenv("MODEL_NAME")

# --- 专业建议：增加一个简单的检查逻辑 ---
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("❌ 错误: 未在 .env 文件中找到 OPENAI_API_KEY。请检查文件内容！")

# 在初始化前先验证一下
Config.validate()

# 初始化客户端
client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)