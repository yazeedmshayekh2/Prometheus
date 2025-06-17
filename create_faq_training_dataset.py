#!/usr/bin/env python3
"""
Script to create training dataset for fine-tuning FAQ embedding model
Using Qwen model to generate alternative questions for better training data
"""

import os
import json
import pyodbc
import pandas as pd
import torch
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import random
from itertools import combinations
import re
from pathlib import Path

# Import Qwen model components (assuming they're available in the system)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QWEN_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Qwen model functionality will be limited.")
    QWEN_AVAILABLE = False

class QwenQuestionGenerator:
    """Wrapper class for Qwen model to generate alternative questions"""
    
    def __init__(self, model_id: str = "Qwen/Qwen3-8B-AWQ"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen model"""
        if not QWEN_AVAILABLE:
            print("⚠️ Qwen model not available. Using fallback question generation.")
            return
            
        try:
            print(f"🔄 Loading Qwen model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                cache_dir='./hf_cache'
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir='./hf_cache'
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
                
            print(f"✅ Qwen model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            print("Falling back to rule-based question generation")
            self.model = None
            self.tokenizer = None

    def generate_alternative_questions(self, original_question: str, language: str = 'en', num_variations: int = 3) -> List[Tuple[str, float]]:
        """
        Generate alternative questions using Qwen model
        
        Args:
            original_question: The original question from the database
            language: Language of the question ('en' or 'ar')
            num_variations: Number of alternative questions to generate
            
        Returns:
            List of tuples (alternative_question, similarity_score)
        """
        
        if not self.model or not self.tokenizer:
            return self._fallback_question_generation(original_question, language, num_variations)
        
        try:
            # Create language-specific prompts
            if language == 'en':
                prompt = f"""You are an expert in generating alternative ways to ask insurance-related questions. 
Given an original insurance question, generate {num_variations} different ways to ask the same question while maintaining the same meaning and context.

Original Question: {original_question}

Please generate {num_variations} alternative questions that:
1. Keep the same meaning and intent
2. Use different wording and sentence structure
3. Are natural and commonly used by customers
4. Maintain insurance domain terminology appropriately

Format your response as a numbered list:
1. [Alternative question 1]
2. [Alternative question 2]
3. [Alternative question 3]

Alternative Questions:"""

            else:  # Arabic
                prompt = f"""أنت خبير في إنشاء طرق بديلة لطرح الأسئلة المتعلقة بالتأمين.
بناءً على السؤال الأصلي للتأمين، قم بإنشاء {num_variations} طرق مختلفة لطرح نفس السؤال مع الحفاظ على نفس المعنى والسياق.

السؤال الأصلي: {original_question}

يرجى إنشاء {num_variations} أسئلة بديلة تتميز بما يلي:
1. الحفاظ على نفس المعنى والهدف
2. استخدام صياغة وبنية جملة مختلفة
3. طبيعية وشائعة الاستخدام من قبل العملاء
4. الحفاظ على مصطلحات مجال التأمين بشكل مناسب

اكتب إجابتك كقائمة مرقمة:
1. [السؤال البديل 1]
2. [السؤال البديل 2]  
3. [السؤال البديل 3]

الأسئلة البديلة:"""

            # Generate response using Qwen
            response = self._generate_text(prompt, max_tokens=512, temperature=0.7)
            
            # Parse the generated questions
            alternative_questions = self._parse_generated_questions(response, num_variations)
            
            # Assign similarity scores (high for Qwen-generated alternatives)
            scored_questions = []
            for i, alt_q in enumerate(alternative_questions):
                # Higher score for earlier (presumably better) generations
                score = 0.95 - (i * 0.05)  # 0.95, 0.90, 0.85, etc.
                scored_questions.append((alt_q, score))
            
            return scored_questions
            
        except Exception as e:
            print(f"⚠️ Error generating questions with Qwen: {e}")
            return self._fallback_question_generation(original_question, language, num_variations)

    def _generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using Qwen model"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response (remove the input prompt)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response

    def _parse_generated_questions(self, response: str, expected_count: int) -> List[str]:
        """Parse generated questions from Qwen response"""
        
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered list format
            if re.match(r'^\d+\.', line):
                # Remove the number and extract the question
                question = re.sub(r'^\d+\.\s*', '', line).strip()
                if question:
                    questions.append(question)
        
        # If parsing failed, try to extract any complete sentences
        if len(questions) == 0:
            sentences = [s.strip() for s in response.split('.') if s.strip() and '?' in s]
            questions = sentences[:expected_count]
        
        # Ensure we have the expected number of questions
        return questions[:expected_count]

    def _fallback_question_generation(self, original_question: str, language: str, num_variations: int) -> List[Tuple[str, float]]:
        """Fallback method when Qwen model is not available"""
        
        variations = []
        
        if language == 'en':
            # Insurance-specific question patterns
            patterns = [
                ("What is", "Can you explain"),
                ("How do I", "What's the process to"),
                ("Can I", "Am I able to"),
                ("What are the", "Tell me about the"),
                ("How much", "What is the cost of"),
                ("covered", "insured"),
                ("policy", "insurance plan"),
                ("claim", "compensation request"),
                ("premium", "insurance payment"),
            ]
            
            base_question = original_question
            for original, replacement in patterns[:num_variations]:
                if original.lower() in base_question.lower():
                    new_question = base_question.replace(original, replacement)
                    if new_question != base_question and new_question not in variations:
                        variations.append(new_question)
                        break
        
        # Fill remaining variations with slight modifications
        while len(variations) < num_variations:
            if language == 'en':
                prefixes = ["Could you please tell me", "I would like to know", "Can you clarify"]
                if len(variations) < len(prefixes):
                    new_q = f"{prefixes[len(variations)]}: {original_question.lower()}"
                    variations.append(new_q)
                else:
                    break
            else:
                break
        
        # Assign scores (lower for fallback method)
        scored_variations = [(var, 0.85 - i * 0.05) for i, var in enumerate(variations)]
        return scored_variations

class FAQDatasetGenerator:
    def __init__(self):
        self.db_server = os.getenv('DB_SERVER', '192.168.3.120')
        self.db_name = os.getenv('DB_NAME', 'agencyDB_Live')
        self.db_user = os.getenv('DB_USER', 'sa')
        self.db_password = os.getenv('DB_PASSWORD', 'P@ssw0rdSQL')
        
        self.conn_str = (
            'DRIVER={ODBC Driver 18 for SQL Server};'
            f'SERVER={self.db_server};'
            f'DATABASE={self.db_name};'
            f'UID={self.db_user};'
            f'PWD={self.db_password};'
            'TrustServerCertificate=yes;'
            'Encrypt=no;'
        )
        
        # Initialize Qwen question generator
        self.question_generator = QwenQuestionGenerator()

    def extract_faq_data(self) -> List[Dict]:
        """Extract FAQ data from database"""
        try:
            connection = pyodbc.connect(self.conn_str)
            cursor = connection.cursor()
            
            query = """
            SELECT q.ID,
                q.CategoryID,
                q.QuestionEN,
                q.QuestionAR,
                a.AnswerEN,
                a.AnswerAR
            FROM agencyDB_Live.dbo.tblFAQQuestions AS q
                    LEFT JOIN agencyDB_Live.dbo.tblFAQAnswer AS a
                            ON q.ID = a.QuestionID
            WHERE q.isDeleted = 0
                AND q.isVisible = 1
                AND (a.isDeleted = 0 AND a.isVisible = 1)
                AND (q.CategoryID >= 10 AND q.CategoryID <= 20)
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            faq_data = []
            for row in results:
                entry = {
                    'id': row[0],
                    'category_id': row[1],
                    'question_en': row[2] if row[2] else '',
                    'question_ar': row[3] if row[3] else '',
                    'answer_en': row[4] if row[4] else '',
                    'answer_ar': row[5] if row[5] else ''
                }
                
                # Only include entries with questions (we'll use questions for training)
                if entry['question_en'] or entry['question_ar']:
                    faq_data.append(entry)
            
            cursor.close()
            connection.close()
            
            print(f"✅ Extracted {len(faq_data)} FAQ entries from database")
            return faq_data
            
        except Exception as e:
            print(f"❌ Error extracting FAQ data: {e}")
            return []

    def create_question_pairs_with_qwen(self, faq_data: List[Dict]) -> List[Tuple[str, str, float]]:
        """Create training pairs using Qwen-generated alternative questions"""
        
        print("🔄 Generating alternative questions using Qwen model...")
        question_pairs = []
        
        for i, faq in enumerate(faq_data):
            print(f"Processing FAQ {i+1}/{len(faq_data)}: ID {faq['id']}")
            
            # Process English questions
            if faq['question_en'] and faq['question_en'].strip():
                original_question = faq['question_en'].strip()
                
                # Generate alternative questions using Qwen
                alternatives = self.question_generator.generate_alternative_questions(
                    original_question, 
                    language='en', 
                    num_variations=3
                )
                
                print(f"✅ Generated {alternatives} alternative questions using Qwen")
                
                # Create pairs: (original_question, alternative_question, similarity_score)
                for alt_question, score in alternatives:
                    if alt_question and alt_question.strip():
                        question_pairs.append((original_question, alt_question.strip(), score))
            
            # Process Arabic questions
            if faq['question_ar'] and faq['question_ar'].strip():
                original_question = faq['question_ar'].strip()
                
                # Generate alternative questions using Qwen
                alternatives = self.question_generator.generate_alternative_questions(
                    original_question, 
                    language='ar', 
                    num_variations=3
                )
                
                # Create pairs: (original_question, alternative_question, similarity_score)
                for alt_question, score in alternatives:
                    if alt_question and alt_question.strip():
                        question_pairs.append((original_question, alt_question.strip(), score))
        
        print(f"✅ Generated {len(question_pairs)} question pairs using Qwen")
        return question_pairs

    def create_negative_pairs(self, faq_data: List[Dict], num_negatives: int = None) -> List[Tuple[str, str, float]]:
        """Create negative pairs (dissimilar questions from different topics)"""
        
        negative_pairs = []
        questions = []
        
        # Collect all questions
        for faq in faq_data:
            if faq['question_en']:
                questions.append(faq['question_en'])
            if faq['question_ar']:
                questions.append(faq['question_ar'])
        
        # Group questions by insurance topics
        insurance_topics = {
            'claim': ['claim', 'compensation', 'settlement', 'payout', 'reimbursement'],
            'coverage': ['cover', 'benefit', 'protection', 'insured', 'policy'],
            'premium': ['premium', 'payment', 'cost', 'fee', 'price'],
            'deductible': ['deductible', 'excess', 'out-of-pocket'],
            'renewal': ['renewal', 'renew', 'extend', 'continue']
        }
        
        topic_questions = {topic: [] for topic in insurance_topics.keys()}
        
        # Categorize questions by topic
        for question in questions:
            for topic, keywords in insurance_topics.items():
                if any(keyword.lower() in question.lower() for keyword in keywords):
                    topic_questions[topic].append(question)
                    break
        
        # Create negative pairs between different topics
        topics = list(topic_questions.keys())
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                q1_list = topic_questions[topic1]
                q2_list = topic_questions[topic2]
                
                if q1_list and q2_list:
                    # Create a few negative pairs between different topics
                    for _ in range(min(5, len(q1_list), len(q2_list))):
                        q1 = random.choice(q1_list)
                        q2 = random.choice(q2_list)
                        negative_pairs.append((q1, q2, 0.2))  # Low similarity score
        
        # Add some random negative pairs
        if num_negatives is None:
            num_negatives = min(len(questions) // 4, 200)  # Limit negative pairs
        
        random_negatives = 0
        while random_negatives < num_negatives and len(questions) > 1:
            q1 = random.choice(questions)
            q2 = random.choice(questions)
            if q1 != q2:  # Ensure they're different
                negative_pairs.append((q1, q2, 0.1))  # Very low similarity
                random_negatives += 1
        
        print(f"✅ Created {len(negative_pairs)} negative pairs")
        return negative_pairs

    def create_training_dataset(self, output_dir: str = "faq_training_data"):
        """Create complete training dataset for FAQ embedding fine-tuning"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract FAQ data
        print("📊 Extracting FAQ data from database...")
        faq_data = self.extract_faq_data()
        if not faq_data:
            print("❌ No FAQ data available for training")
            return
        
        # Save raw FAQ data
        with open(f"{output_dir}/raw_faq_data.json", 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        
        # Create question pairs using Qwen
        print("🤖 Creating question pairs using Qwen model...")
        positive_pairs = self.create_question_pairs_with_qwen(faq_data)
        
        # Create negative pairs for contrast
        print("🔄 Creating negative pairs...")
        negative_pairs = self.create_negative_pairs(faq_data)
        
        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Convert to training format
        training_data = []
        for text1, text2, score in all_pairs:
            training_data.append({
                'text1': text1,
                'text2': text2,
                'similarity_score': score
            })
        
        # Split into train/val/test
        train_data, temp_data = train_test_split(training_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save datasets
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            # JSON format
            with open(f"{output_dir}/{split_name}.json", 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # CSV format for easy inspection
            df = pd.DataFrame(split_data)
            df.to_csv(f"{output_dir}/{split_name}.csv", index=False, encoding='utf-8')
        
        print(f"\n🎉 Dataset created successfully!")
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print(f"   Test samples: {len(test_data)}")
        print(f"   Total samples: {len(training_data)}")
        print(f"   Positive pairs (Qwen-generated): {len(positive_pairs)}")
        print(f"   Negative pairs: {len(negative_pairs)}")
        
        # Create dataset statistics
        stats = {
            'total_faqs': len(faq_data),
            'total_training_pairs': len(training_data),
            'positive_pairs_qwen': len(positive_pairs),
            'negative_pairs': len(negative_pairs),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'generation_method': 'qwen_model',
            'model_used': self.question_generator.model_id if self.question_generator.model else 'fallback'
        }
        
        with open(f"{output_dir}/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save a sample for inspection
        sample_data = training_data[:10]  # First 10 samples
        with open(f"{output_dir}/sample_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 All files saved to: {output_dir}")
        print(f"📝 Check sample_pairs.json to see example training pairs")
        
        return output_dir

if __name__ == "__main__":
    print("🚀 Starting FAQ Training Dataset Generation with Qwen Model")
    print("=" * 60)
    
    generator = FAQDatasetGenerator()
    dataset_dir = generator.create_training_dataset()
    
    if dataset_dir:
        print(f"\n✅ Training dataset saved to: {dataset_dir}")
        print("\n🔄 Next steps:")
        print("1. Inspect the generated sample_pairs.json file")
        print("2. Run: python finetune_faq_embedding.py")
        print("3. After training, run: python integrate_finetuned_model.py")
    else:
        print("\n❌ Dataset generation failed") 