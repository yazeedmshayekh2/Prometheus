import sqlite3
from datetime import datetime
import os
import uuid  # Add UUID import

class AuthDB:
    def __init__(self):
        # Create the database directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        self.db_path = 'data/auth.db'
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            conn.commit()

    def create_user(self, user_id, email, name, hashed_password):
        """Create a new user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (id, email, name, hashed_password) VALUES (?, ?, ?, ?)',
                    (user_id, email, name, hashed_password)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def get_user_by_email(self, email):
        """Get user by email"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            if user:
                return {
                    'id': user[0],
                    'email': user[1],
                    'name': user[2],
                    'hashed_password': user[3]
                }
            return None

    def create_conversation(self, conversation_id, user_id):
        """Create a new conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (id, user_id) VALUES (?, ?)',
                (conversation_id, user_id)
            )
            conn.commit()

    def get_user_conversations(self, user_id):
        """Get all conversations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, GROUP_CONCAT(json_object(
                    'role', m.role,
                    'content', m.content
                )) as messages
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            ''', (user_id,))
            conversations = cursor.fetchall()
            return [{
                'id': conv['id'],
                'user_id': conv['user_id'],
                'created_at': conv['created_at'],
                'updated_at': conv['updated_at'],
                'messages': eval(conv['messages']) if conv['messages'] else []
            } for conv in conversations]

    def get_conversation(self, conversation_id, user_id):
        """Get a specific conversation and its messages"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, json_group_array(json_object(
                    'role', m.role,
                    'content', m.content
                )) as messages
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.id = ? AND c.user_id = ?
                GROUP BY c.id
            ''', (conversation_id, user_id))
            conv = cursor.fetchone()
            if conv:
                return {
                    'id': conv['id'],
                    'user_id': conv['user_id'],
                    'created_at': conv['created_at'],
                    'updated_at': conv['updated_at'],
                    'messages': eval(conv['messages']) if conv['messages'] else []
                }
            return None

    def update_conversation(self, conversation_id, messages):
        """Update a conversation's messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete existing messages
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            # Insert new messages
            for msg in messages:
                cursor.execute(
                    'INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)',
                    (str(uuid.uuid4()), conversation_id, msg['role'], msg['content'])
                )
            # Update conversation timestamp
            cursor.execute(
                'UPDATE conversations SET updated_at = ? WHERE id = ?',
                (datetime.utcnow().isoformat(), conversation_id)
            )
            conn.commit() 