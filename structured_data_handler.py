from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import re

class StructuredDataResult(BaseModel):
    field_name: str
    value: Union[str, float, int]
    confidence: float
    source: str
    timestamp: Optional[datetime] = None

class StructuredDataHandler:
    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}
        
    def get_structured_data(self, policy_id: str, fields: List[str]) -> List[StructuredDataResult]:
        """
        Retrieve structured data from database for specific fields
        """
        results = []
        try:
            # Check cache first
            cache_key = f"{policy_id}:{','.join(fields)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Query database
            query = """
            SELECT field_name, field_value, last_updated
            FROM policy_structured_data
            WHERE policy_id = ? AND field_name IN ({})
            """.format(','.join(['?' for _ in fields]))
            
            df = pd.read_sql(query, self.db.connection, params=[policy_id] + fields)
            
            for _, row in df.iterrows():
                results.append(
                    StructuredDataResult(
                        field_name=row['field_name'],
                        value=row['field_value'],
                        confidence=1.0,  # High confidence for structured data
                        source='database',
                        timestamp=row['last_updated']
                    )
                )
            
            # Cache results
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            print(f"Error retrieving structured data: {str(e)}")
            return []
    
    def extract_structured_fields(self, text: str) -> Dict[str, StructuredDataResult]:
        """
        Extract structured fields from text using regex patterns
        """
        patterns = {
            'annual_limit': (
                r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:annual|yearly|per year|maximum)',
                lambda x: float(re.sub(r'[^\d.]', '', x))
            ),
            'copayment': (
                r'\$?\d+(?:\.\d{2})?\s*(?:copay|copayment)',
                lambda x: float(re.sub(r'[^\d.]', '', x))
            ),
            'coinsurance': (
                r'\d+%\s*(?:coinsurance|cost sharing)',
                lambda x: float(re.sub(r'[^\d.]', '', x))
            ),
            'deductible': (
                r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:deductible)',
                lambda x: float(re.sub(r'[^\d.]', '', x))
            ),
            'waiting_period': (
                r'(\d+)\s*(?:day|week|month|year)s?\s*waiting\s*period',
                lambda x: int(re.search(r'\d+', x).group())
            )
        }
        
        results = {}
        for field, (pattern, converter) in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = converter(match.group())
                results[field] = StructuredDataResult(
                    field_name=field,
                    value=value,
                    confidence=0.9,  # High confidence for regex matches
                    source='text_extraction',
                    timestamp=datetime.now()
                )
        
        return results
    
    def merge_structured_data(self, 
                            db_results: List[StructuredDataResult],
                            extracted_results: Dict[str, StructuredDataResult]) -> Dict[str, StructuredDataResult]:
        """
        Merge structured data from database and text extraction
        """
        merged = {}
        
        # First add database results (higher priority)
        for result in db_results:
            merged[result.field_name] = result
        
        # Add extracted results if not in database
        for field, result in extracted_results.items():
            if field not in merged:
                merged[field] = result
        
        return merged
    
    def get_relevant_structured_data(self, question_type: str, policy_id: str, text: str) -> Dict[str, StructuredDataResult]:
        """
        Get relevant structured data based on question type
        """
        field_mapping = {
            'NUMERICAL': ['annual_limit', 'copayment', 'coinsurance', 'deductible'],
            'TEMPORAL': ['waiting_period', 'effective_date', 'termination_date'],
            'COVERAGE': ['covered_services', 'exclusions', 'limitations'],
            'POLICY_SPECIFIC': ['policy_type', 'network_type', 'plan_level']
        }
        
        relevant_fields = field_mapping.get(question_type, [])
        if not relevant_fields:
            return {}
        
        # Get data from both sources
        db_results = self.get_structured_data(policy_id, relevant_fields)
        extracted_results = self.extract_structured_fields(text)
        
        # Merge results
        return self.merge_structured_data(db_results, extracted_results) 