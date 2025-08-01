"""
Utility functions for OpenBioGen AI
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class GeneValidator:
    """Validate gene symbols and disease names"""
    
    @staticmethod
    def validate_gene_symbol(gene: str) -> bool:
        """Validate gene symbol format"""
        if not gene or len(gene.strip()) == 0:
            return False
        
        # Basic gene symbol validation (alphanumeric, hyphens, underscores)
        pattern = r'^[A-Za-z0-9_-]+$'
        return bool(re.match(pattern, gene.strip()))
    
    @staticmethod
    def validate_disease_name(disease: str) -> bool:
        """Validate disease name format"""
        if not disease or len(disease.strip()) < 2:
            return False
        
        # Allow letters, numbers, spaces, hyphens, apostrophes
        pattern = r'^[A-Za-z0-9\s\-\']+$'
        return bool(re.match(pattern, disease.strip()))

class ResultsProcessor:
    """Process and format prediction results"""
    
    @staticmethod
    def format_prediction_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Format prediction result for display"""
        formatted = {
            'gene': result.get('gene', '').upper(),
            'disease': result.get('disease', '').title(),
            'prediction': result.get('prediction', ''),
            'confidence': result.get('confidence', 'Low'),
            'sources_count': len(result.get('sources', [])),
            'sources': result.get('sources', []),
            'timestamp': datetime.now().isoformat()
        }
        return formatted
    
    @staticmethod
    def export_results_to_csv(results: List[Dict[str, Any]], filename: str = None) -> str:
        """Export results to CSV format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gene_disease_predictions_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        
        # Flatten sources list for CSV
        df['sources'] = df['sources'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        
        csv_content = df.to_csv(index=False)
        return csv_content
    
    @staticmethod
    def calculate_batch_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for batch predictions"""
        if not results:
            return {}
        
        total = len(results)
        confidence_counts = {}
        
        for result in results:
            confidence = result.get('confidence', 'Low')
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        stats = {
            'total_predictions': total,
            'confidence_distribution': confidence_counts,
            'high_confidence_percentage': round((confidence_counts.get('High', 0) / total) * 100, 2),
            'sources_found_percentage': round(
                (sum(1 for r in results if r.get('sources')) / total) * 100, 2
            )
        }
        
        return stats

class LiteratureProcessor:
    """Process scientific literature content"""
    
    @staticmethod
    def extract_key_terms(text: str) -> List[str]:
        """Extract key biological terms from text"""
        # Common biological terms patterns
        patterns = [
            r'\b[A-Z][A-Z0-9]{2,}\b',  # Gene symbols (e.g., BRCA1, TP53)
            r'\b\w+ase\b',  # Enzymes ending in -ase
            r'\b\w+oma\b',  # Tumors ending in -oma
            r'\bprotein\s+\w+\b',  # Protein names
            r'\bpathway\b',  # Pathway mentions
            r'\bmutation\b',  # Mutation mentions
        ]
        
        key_terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_terms.update([match.lower() for match in matches])
        
        return list(key_terms)
    
    @staticmethod
    def summarize_document(content: str, max_length: int = 200) -> str:
        """Create a summary of document content"""
        if len(content) <= max_length:
            return content
        
        # Simple extractive summarization - take first and last sentences
        sentences = content.split('. ')
        if len(sentences) <= 2:
            return content[:max_length] + "..."
        
        summary = sentences[0] + '. ' + sentences[-1]
        if len(summary) > max_length:
            return content[:max_length] + "..."
        
        return summary

class ConfigManager:
    """Manage application configuration"""
    
    DEFAULT_CONFIG = {
        'max_search_results': 5,
        'confidence_threshold': 0.7,
        'default_llm_model': 'microsoft/DialoGPT-medium',
        'search_domains': [
            'pubmed.ncbi.nlm.nih.gov',
            'omim.org',
            'genecards.org'
        ],
        'max_context_length': 2000,
        'temperature': 0.7
    }
    
    @classmethod
    def load_config(cls, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config = cls.DEFAULT_CONFIG.copy()
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], config_file: str) -> bool:
        """Save configuration to file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Could not save config file {config_file}: {e}")
            return False

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def create_sample_data() -> pd.DataFrame:
    """Create sample gene-disease pairs for testing"""
    sample_data = [
        ('BRCA1', 'breast cancer'),
        ('BRCA2', 'ovarian cancer'),
        ('TP53', 'Li-Fraumeni syndrome'),
        ('APOE', "Alzheimer's disease"),
        ('CFTR', 'cystic fibrosis'),
        ('HTT', "Huntington's disease"),
        ('LRRK2', "Parkinson's disease"),
        ('SOD1', 'amyotrophic lateral sclerosis'),
        ('PSEN1', "Alzheimer's disease"),
        ('MAPT', 'frontotemporal dementia')
    ]
    
    df = pd.DataFrame(sample_data, columns=['gene', 'disease'])
    return df
