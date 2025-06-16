import sqlite3
from datetime import datetime, timedelta
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
                    email TEXT UNIQUE,
                    name TEXT,
                    hashed_password TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create password reset tokens table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    token TEXT UNIQUE,
                    expires_at TIMESTAMP,
                    used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    archived BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Add archived column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE conversations ADD COLUMN archived BOOLEAN DEFAULT FALSE')
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Create messages table with additional state columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')

            # Create conversation state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_state (
                    conversation_id TEXT PRIMARY KEY,
                    contractor_name TEXT,
                    expiry_date TEXT,
                    beneficiary_count TEXT,
                    national_id TEXT,
                    suggested_questions TEXT,
                    is_national_id_confirmed BOOLEAN,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
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

    def get_user_by_id(self, user_id):
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if user:
                return {
                    'id': user[0],
                    'email': user[1],
                    'name': user[2],
                    'hashed_password': user[3]
                }
            return None

    def create_password_reset_token(self, user_id, token, expires_at):
        """Create a password reset token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First, mark any existing unused tokens as used
                cursor.execute(
                    'UPDATE password_reset_tokens SET used = TRUE WHERE user_id = ? AND used = FALSE',
                    (user_id,)
                )
                
                # Create new token
                token_id = str(uuid.uuid4())
                cursor.execute(
                    'INSERT INTO password_reset_tokens (id, user_id, token, expires_at) VALUES (?, ?, ?, ?)',
                    (token_id, user_id, token, expires_at)
                )
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def get_valid_reset_token(self, token):
        """Get a valid reset token (not used and not expired)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT rt.user_id, u.email, u.name 
                FROM password_reset_tokens rt
                JOIN users u ON rt.user_id = u.id
                WHERE rt.token = ? AND rt.used = FALSE AND rt.expires_at > datetime('now')
            ''', (token,))
            result = cursor.fetchone()
            if result:
                return {
                    'user_id': result[0],
                    'email': result[1],
                    'name': result[2]
                }
            return None

    def use_reset_token(self, token):
        """Mark a reset token as used"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE password_reset_tokens SET used = TRUE WHERE token = ?',
                (token,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_user_password(self, user_id, new_hashed_password):
        """Update user password"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET hashed_password = ? WHERE id = ?',
                    (new_hashed_password, user_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False

    def cleanup_expired_tokens(self):
        """Remove expired password reset tokens"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM password_reset_tokens WHERE expires_at < datetime("now")'
            )
            conn.commit()

    def create_conversation(self, conversation_id, user_id):
        """Create a new conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (id, user_id) VALUES (?, ?)',
                (conversation_id, user_id)
            )
            conn.commit()

    def get_user_conversations(self, user_id, include_archived=False):
        """Get all conversations for a user, optionally including archived ones"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query based on whether to include archived conversations
            if include_archived:
                query = '''
                SELECT c.*, GROUP_CONCAT(json_object(
                    'role', m.role,
                    'content', m.content
                )) as messages
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = ?
                    GROUP BY c.id
                    ORDER BY c.archived ASC, c.updated_at DESC
                '''
            else:
                query = '''
                    SELECT c.*, GROUP_CONCAT(json_object(
                        'role', m.role,
                        'content', m.content
                    )) as messages
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.user_id = ? AND c.archived = FALSE
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                '''
            
            cursor.execute(query, (user_id,))
            conversations = cursor.fetchall()
            return [{
                'id': conv['id'],
                'user_id': conv['user_id'],
                'archived': bool(conv['archived']),
                'created_at': conv['created_at'],
                'updated_at': conv['updated_at'],
                'messages': eval(conv['messages']) if conv['messages'] else []
            } for conv in conversations]

    def get_conversation(self, conversation_id, user_id):
        """Get a conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get conversation
            cursor.execute(
                'SELECT id, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?',
                (conversation_id, user_id)
            )
            conversation = cursor.fetchone()
            
            if not conversation:
                return None
            
            # Get messages
            cursor.execute(
                'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at',
                (conversation_id,)
            )
            messages = [{'role': role, 'content': content} for role, content in cursor.fetchall()]
            
            # Get conversation state
            cursor.execute(
                'SELECT contractor_name, expiry_date, beneficiary_count, national_id, suggested_questions, is_national_id_confirmed FROM conversation_state WHERE conversation_id = ?',
                (conversation_id,)
            )
            state = cursor.fetchone()
            
            return {
                'id': conversation[0],
                'messages': messages,
                'created_at': conversation[1],
                'updated_at': conversation[2],
                'userInfo': {
                    'contractorName': state[0] if state else '-',
                    'expiryDate': state[1] if state else '-',
                    'beneficiaryCount': state[2] if state else '-',
                    'nationalId': state[3] if state else ''
                } if state else None,
                'suggestedQuestions': state[4] if state else '',
                'isNationalIdConfirmed': bool(state[5]) if state else False
            }

    def update_conversation(self, conversation_id, messages, user_info=None, suggested_questions=None, is_national_id_confirmed=None):
        """Update a conversation's messages and state"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update conversation timestamp
            cursor.execute(
                'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (conversation_id,)
            )
            
            # Delete existing messages
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            
            # Insert new messages
            for msg in messages:
                msg_id = str(uuid.uuid4())
                cursor.execute(
                    'INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)',
                    (msg_id, conversation_id, msg['role'], msg['content'])
                )

            # Update conversation state
            if user_info is not None:
                cursor.execute('DELETE FROM conversation_state WHERE conversation_id = ?', (conversation_id,))
                cursor.execute('''
                    INSERT INTO conversation_state (
                        conversation_id, contractor_name, expiry_date, 
                        beneficiary_count, national_id, suggested_questions, 
                        is_national_id_confirmed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id,
                    user_info.get('contractorName'),
                    user_info.get('expiryDate'),
                    user_info.get('beneficiaryCount'),
                    user_info.get('nationalId'),
                    suggested_questions,
                    is_national_id_confirmed
                )) 
            
            conn.commit()

    def delete_conversation(self, conversation_id, user_id):
        """Delete a conversation and all its associated data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Verify conversation belongs to user
                cursor.execute(
                    'SELECT id FROM conversations WHERE id = ? AND user_id = ?',
                    (conversation_id, user_id)
                )
                if not cursor.fetchone():
                    return False
                
                # Delete conversation state
                cursor.execute('DELETE FROM conversation_state WHERE conversation_id = ?', (conversation_id,))
                
                # Delete messages
                cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
                
                # Delete conversation
                cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
                
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def archive_conversation(self, conversation_id, user_id, archived=True):
        """Archive or unarchive a conversation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Verify conversation belongs to user and update archived status
                cursor.execute(
                    'UPDATE conversations SET archived = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?',
                    (archived, conversation_id, user_id)
                )
                
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False 