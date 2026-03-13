"""
这个文件专门负责与 SQLite 打交道。
实现了记录历史对话记录，同时用户能够选择继续对话。
它能确保即使程序中途崩溃，对话和摘要也能安全地存在硬盘里。
"""

import sqlite3
import json
import uuid
from datetime import datetime
from config import Config

class ChatDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # 会话表：管理元数据
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
        ''')
        # 消息表：记录每一条对话，包括工具调用
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                tokens INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 摘要表：记录每个 session 的背景
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                summary_content TEXT,
                last_msg_id INTEGER
            )
        ''')
        self.conn.commit()


    # --- Session 管理 ---
    def create_session(self, title="新会话"):
        """首次进入对话时，先创建一个新会话记录在 sessions 表中"""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
            (session_id, title, now, now)
        )
        self.conn.commit()
        return session_id

    def update_session_title(self, session_id, title):
        """进行一轮对话后调用一次LLM生成本次会话的标题并记录"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE sessions SET title = ? WHERE session_id = ?', (title, session_id))
        self.conn.commit()

    def get_session_title(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT title FROM sessions WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        return result[0] if result else "未知会话"

    def get_recent_sessions(self, limit=10):
        """获取上限为 limit 数量的历史对话，为用户输入/resume后检索使用，按照活跃时间取最近的对话"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT session_id, title, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?', (limit,))
        return cursor.fetchall()


    # --- Message 管理 ---
    def save_message(self, session_id, role, content=None, tool_calls=None, tool_call_id=None, tokens=0):
        """为本次会话储存每一条消息"""
        cursor = self.conn.cursor()
        # 将 tool_calls 序列化为 JSON 字符串存储
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, role, content, tool_calls_json, tool_call_id, tokens))
        self.conn.commit()

        # 更新会话活跃时间
        cursor.execute('UPDATE sessions SET updated_at = ? WHERE session_id = ?', 
                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id))
        self.conn.commit()
        return cursor.lastrowid

    def get_messages_after(self, session_id, last_msg_id=0):
        """获取指定 session ID 在上次摘要后的所有消息"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT role, content, tool_calls, tool_call_id FROM messages 
            WHERE session_id = ? AND id > ?
            ORDER BY id ASC
        ''', (session_id, last_msg_id))
        rows = cursor.fetchall()
        
        messages = []
        for r in rows:
            m = {"role": r[0], "content": r[1]}
            if r[2]: m["tool_calls"] = json.loads(r[2])
            if r[3]: m["tool_call_id"] = r[3]
            messages.append(m)
        return messages


    # --- Summary 管理 ---
    def get_summary(self, session_id):
        """获取指定 session ID 的摘要"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT summary_content, last_msg_id FROM summaries WHERE session_id = ?', (session_id,))
        return cursor.fetchone()

    def update_summary(self, session_id, content, last_msg_id):
        """更新指定 session ID 的摘要"""
        cursor = self.conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO summaries (session_id, summary_content, last_msg_id) VALUES (?, ?, ?)',
                       (session_id, content, last_msg_id))
        self.conn.commit()

# 初始化全局数据库实例
db = ChatDatabase()