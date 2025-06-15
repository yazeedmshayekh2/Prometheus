"""
Database connection module for SQL Server
"""

import os
import logging
import pyodbc

# Configure logging
logger = logging.getLogger(__name__)

def connect_to_database():
    """
    Create a connection to the SQL Server database.
    
    Returns:
        pyodbc.Connection: Database connection object
        
    Raises:
        pyodbc.Error: If connection fails
    """
    # Get database connection parameters from environment variables
    db_server = os.getenv('DB_SERVER', '192.168.3.120')
    db_name = os.getenv('DB_NAME', 'agencyDB_Live')
    db_user = os.getenv('DB_USER', 'sa')
    db_password = os.getenv('DB_PASSWORD', 'P@ssw0rdSQL')
    
    conn_str = (
        'DRIVER={ODBC Driver 18 for SQL Server};'
        f'SERVER={db_server};'
        f'DATABASE={db_name};'
        f'UID={db_user};'
        f'PWD={db_password};'
        'TrustServerCertificate=yes;'
        'Encrypt=no;'  
    )
   
    try:
        connection = pyodbc.connect(conn_str)
        logger.info("Successfully connected to the database!")
        return connection
    except pyodbc.Error as e:
        logger.error(f"Error connecting to the database: {e}")
        raise 

def get_faq_answer(question: str) -> str:
    """
    Check if the question exists in the FAQ database and return the exact answer.
    
    Args:
        question (str): The user's question to search for
        
    Returns:
        str: The FAQ answer if found, None if not found
    """
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        
        # Query to find exact match for the question (both English and Arabic)
        query = """
        SELECT 
            a.AnswerEN AS Answer_English, 
            a.AnswerAR AS Answer_Arabic
        FROM agencyDB_Live.dbo.tblFAQQuestions AS q
        LEFT JOIN agencyDB_Live.dbo.tblFAQAnswer AS a
            ON q.ID = a.QuestionID
        WHERE q.isDeleted = 0 AND q.isVisible = 1
          AND (a.isDeleted = 0 AND a.isVisible = 1)
          AND (LOWER(TRIM(q.QuestionEN)) = LOWER(TRIM(?)) 
               OR LOWER(TRIM(q.QuestionAR)) = LOWER(TRIM(?)))
        """
        
        cursor.execute(query, (question.strip(), question.strip()))
        result = cursor.fetchone()
        
        if result:
            # Prefer English answer, fallback to Arabic if English is not available
            answer_en = result[0] if result[0] else ""
            answer_ar = result[1] if result[1] else ""
            
            # Return English answer if available, otherwise Arabic
            final_answer = answer_en if answer_en.strip() else answer_ar
            
            if final_answer and final_answer.strip():
                logger.info(f"Found FAQ answer for question: {question[:50]}...")
                return final_answer.strip()
            else:
                logger.info(f"FAQ answer found but empty for question: {question[:50]}...")
                return None
        else:
            logger.info(f"No FAQ answer found for question: {question[:50]}...")
            return None
            
    except pyodbc.Error as e:
        logger.error(f"Error querying FAQ database: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_faq_answer: {e}")
        return None
    finally:
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
        except:
            pass

def search_similar_faq(question: str, similarity_threshold: float = 0.7) -> str:
    """
    Search for similar FAQ questions using fuzzy matching.
    This is a fallback when exact match is not found.
    
    Args:
        question (str): The user's question to search for
        similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
        
    Returns:
        str: The FAQ answer if similar question found, None if not found
    """
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        
        # Get all FAQ questions and answers
        query = """
        SELECT 
            q.QuestionEN AS Question_English, 
            q.QuestionAR AS Question_Arabic,
            a.AnswerEN AS Answer_English, 
            a.AnswerAR AS Answer_Arabic
        FROM agencyDB_Live.dbo.tblFAQQuestions AS q
        LEFT JOIN agencyDB_Live.dbo.tblFAQAnswer AS a
            ON q.ID = a.QuestionID
        WHERE q.isDeleted = 0 AND q.isVisible = 1
          AND (a.isDeleted = 0 AND a.isVisible = 1)
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            return None
        
        # Simple similarity check using common words
        question_lower = question.lower().strip()
        question_words = set(question_lower.split())
        
        best_match = None
        best_score = 0
        
        for row in results:
            question_en = row[0] if row[0] else ""
            question_ar = row[1] if row[1] else ""
            answer_en = row[2] if row[2] else ""
            answer_ar = row[3] if row[3] else ""
            
            # Check similarity against both English and Arabic questions
            for faq_question in [question_en, question_ar]:
                if not faq_question or not faq_question.strip():
                    continue
                    
                faq_question_lower = faq_question.lower().strip()
                faq_words = set(faq_question_lower.split())
                
                # Calculate Jaccard similarity
                intersection = question_words.intersection(faq_words)
                union = question_words.union(faq_words)
                
                if len(union) > 0:
                    similarity = len(intersection) / len(union)
                    
                    if similarity > best_score and similarity >= similarity_threshold:
                        best_score = similarity
                        # Prefer English answer, fallback to Arabic
                        best_match = answer_en.strip() if answer_en.strip() else answer_ar.strip()
        
        if best_match:
            logger.info(f"Found similar FAQ answer with score {best_score:.2f} for question: {question[:50]}...")
            return best_match
        else:
            logger.info(f"No similar FAQ answer found for question: {question[:50]}...")
            return None
            
    except pyodbc.Error as e:
        logger.error(f"Error searching similar FAQ: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in search_similar_faq: {e}")
        return None
    finally:
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
        except:
            pass 