"""
Advanced Memory System for OpenBioGen AI
Implements Semantic, Episodic, and Procedural Memory for intelligent user interactions
"""

import json
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

@dataclass
class MemoryEntry:
    """Base class for memory entries"""
    id: str = ""
    timestamp: datetime = None
    content: Dict[str, Any] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.content is None:
            self.content = {}

@dataclass
class SemanticMemory(MemoryEntry):
    """Semantic memory for factual knowledge"""
    concept: str = ""
    relationships: List[str] = None
    confidence: float = 0.8
    
    def __post_init__(self):
        super().__post_init__()
        if self.relationships is None:
            self.relationships = []

@dataclass
class EpisodicMemory(MemoryEntry):
    """Episodic memory for user interactions and experiences"""
    user_id: str = ""
    context: str = ""
    outcome: str = ""
    emotional_valence: float = 0.0

@dataclass
class ProceduralMemory(MemoryEntry):
    """Procedural memory for learned processes and patterns"""
    procedure_name: str = ""
    steps: List[str] = None
    success_rate: float = 0.0
    optimization_level: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if self.steps is None:
            self.steps = []

class AdvancedMemorySystem:
    """Advanced memory system with semantic, episodic, and procedural memory"""
    
    def __init__(self, memory_dir: str = "memory_storage"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize memory stores
        self.semantic_memory = {}
        self.episodic_memory = deque(maxlen=10000)  # Keep last 10k episodes
        self.procedural_memory = {}
        
        # Initialize databases
        self.db_path = os.path.join(memory_dir, "memory.db")
        self.init_database()
        
        # Initialize semantic search
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.semantic_vectors = None
        self.semantic_concepts = []
        
        # Memory consolidation
        self.consolidation_threshold = 0.7
        self.forgetting_curve_factor = 0.1
        
        # Load existing memories
        self.load_memories()
        
        # Background consolidation thread
        self.consolidation_thread = threading.Thread(target=self._background_consolidation, daemon=True)
        self.consolidation_thread.start()
    
    def init_database(self):
        """Initialize SQLite database for persistent memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Semantic memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id TEXT PRIMARY KEY,
                concept TEXT,
                content TEXT,
                relationships TEXT,
                confidence REAL,
                importance REAL,
                access_count INTEGER,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            )
        ''')
        
        # Episodic memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                context TEXT,
                content TEXT,
                outcome TEXT,
                emotional_valence REAL,
                importance REAL,
                access_count INTEGER,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            )
        ''')
        
        # Procedural memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS procedural_memory (
                id TEXT PRIMARY KEY,
                procedure_name TEXT,
                steps TEXT,
                success_rate REAL,
                optimization_level INTEGER,
                importance REAL,
                access_count INTEGER,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_semantic_memory(self, concept: str, content: Dict[str, Any], 
                            relationships: List[str] = None, confidence: float = 0.8):
        """Store semantic knowledge"""
        memory_id = hashlib.md5(f"{concept}_{str(content)}".encode()).hexdigest()
        
        semantic_mem = SemanticMemory(
            id=memory_id,
            timestamp=datetime.now(),
            content=content,
            concept=concept,
            relationships=relationships or [],
            confidence=confidence
        )
        
        self.semantic_memory[memory_id] = semantic_mem
        self._persist_semantic_memory(semantic_mem)
        self._update_semantic_vectors()
        
        return memory_id
    
    def store_episodic_memory(self, user_id: str, context: str, content: Dict[str, Any],
                            outcome: str, emotional_valence: float = 0.0):
        """Store episodic experience"""
        memory_id = hashlib.md5(f"{user_id}_{context}_{time.time()}".encode()).hexdigest()
        
        episodic_mem = EpisodicMemory(
            id=memory_id,
            timestamp=datetime.now(),
            content=content,
            user_id=user_id,
            context=context,
            outcome=outcome,
            emotional_valence=emotional_valence
        )
        
        self.episodic_memory.append(episodic_mem)
        self._persist_episodic_memory(episodic_mem)
        
        return memory_id
    
    def store_procedural_memory(self, procedure_name: str, steps: List[str],
                              success_rate: float = 0.0, optimization_level: int = 1):
        """Store procedural knowledge"""
        memory_id = hashlib.md5(f"{procedure_name}_{str(steps)}".encode()).hexdigest()
        
        procedural_mem = ProceduralMemory(
            id=memory_id,
            timestamp=datetime.now(),
            content={"steps": steps, "metadata": {}},
            procedure_name=procedure_name,
            steps=steps,
            success_rate=success_rate,
            optimization_level=optimization_level
        )
        
        self.procedural_memory[memory_id] = procedural_mem
        self._persist_procedural_memory(procedural_mem)
        
        return memory_id
    
    def retrieve_semantic_memory(self, query: str, top_k: int = 5) -> List[SemanticMemory]:
        """Retrieve relevant semantic memories"""
        if not self.semantic_vectors is not None or not self.semantic_concepts:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.semantic_vectors).flatten()
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                concept = self.semantic_concepts[idx]
                for mem_id, memory in self.semantic_memory.items():
                    if memory.concept == concept:
                        memory.access_count += 1
                        memory.last_accessed = datetime.now()
                        results.append(memory)
                        break
        
        return results
    
    def retrieve_episodic_memory(self, user_id: str = None, context: str = None,
                               days_back: int = 30) -> List[EpisodicMemory]:
        """Retrieve episodic memories"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        results = []
        for memory in self.episodic_memory:
            if memory.timestamp < cutoff_date:
                continue
            
            if user_id and memory.user_id != user_id:
                continue
            
            if context and context.lower() not in memory.context.lower():
                continue
            
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def retrieve_procedural_memory(self, procedure_name: str = None) -> List[ProceduralMemory]:
        """Retrieve procedural memories"""
        results = []
        
        for memory in self.procedural_memory.values():
            if procedure_name and procedure_name.lower() not in memory.procedure_name.lower():
                continue
            
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
        
        return sorted(results, key=lambda x: x.success_rate, reverse=True)
    
    def get_protein_suggestions(self, query: str) -> List[str]:
        """Get protein suggestions based on semantic memory"""
        # Common proteins and their aliases
        protein_knowledge = {
            'BRCA1': ['breast cancer 1', 'brca1', 'breast cancer gene 1'],
            'BRCA2': ['breast cancer 2', 'brca2', 'breast cancer gene 2'],
            'TP53': ['tumor protein p53', 'tp53', 'p53', 'tumor suppressor p53'],
            'APOE': ['apolipoprotein e', 'apoe', 'apo e'],
            'ATM': ['ataxia telangiectasia mutated', 'atm'],
            'CHEK2': ['checkpoint kinase 2', 'chk2', 'chek2'],
            'RAD51': ['rad51 recombinase', 'rad51'],
            'MDM2': ['mdm2 proto-oncogene', 'mdm2'],
            'RB1': ['retinoblastoma 1', 'rb1', 'retinoblastoma protein'],
            'PTEN': ['phosphatase and tensin homolog', 'pten'],
            'APC': ['adenomatous polyposis coli', 'apc'],
            'MLH1': ['muts homolog 1', 'mlh1'],
            'MSH2': ['muts homolog 2', 'msh2'],
            'CFTR': ['cystic fibrosis transmembrane conductance regulator', 'cftr'],
            'HTT': ['huntingtin', 'htt', 'huntington protein'],
            'SOD1': ['superoxide dismutase 1', 'sod1'],
            'LRRK2': ['leucine rich repeat kinase 2', 'lrrk2'],
            'SNCA': ['synuclein alpha', 'snca', 'alpha-synuclein'],
            'PARK2': ['parkin rbre3 ubiquitin protein ligase', 'park2', 'parkin'],
            'PINK1': ['pten induced kinase 1', 'pink1'],
            'VHL': ['von hippel-lindau tumor suppressor', 'vhl'],
            'NF1': ['neurofibromin 1', 'nf1'],
            'NF2': ['neurofibromin 2', 'nf2'],
            'TSC1': ['tuberous sclerosis 1', 'tsc1'],
            'TSC2': ['tuberous sclerosis 2', 'tsc2']
        }
        
        query_lower = query.lower()
        suggestions = []
        
        # Direct matches
        for protein, aliases in protein_knowledge.items():
            if query_lower in protein.lower() or any(query_lower in alias for alias in aliases):
                suggestions.append(protein)
        
        # Semantic memory search
        semantic_results = self.retrieve_semantic_memory(query, top_k=10)
        for result in semantic_results:
            if 'protein' in result.content or 'gene' in result.content:
                concept = result.concept.upper()
                if concept not in suggestions:
                    suggestions.append(concept)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def learn_from_interaction(self, user_input: str, system_response: str, 
                             success: bool, context: str = "general"):
        """Learn from user interactions"""
        # Store episodic memory
        self.store_episodic_memory(
            user_id="default_user",
            context=context,
            content={
                "user_input": user_input,
                "system_response": system_response,
                "success": success
            },
            outcome="success" if success else "failure",
            emotional_valence=0.5 if success else -0.3
        )
        
        # Update procedural memory
        if context in ["gene_analysis", "protein_search", "network_analysis"]:
            procedure_name = f"{context}_procedure"
            existing_procedures = self.retrieve_procedural_memory(procedure_name)
            
            if existing_procedures:
                # Update success rate
                proc = existing_procedures[0]
                old_rate = proc.success_rate
                new_rate = (old_rate + (1.0 if success else 0.0)) / 2
                proc.success_rate = new_rate
                proc.optimization_level += 1 if success else 0
            else:
                # Create new procedure
                self.store_procedural_memory(
                    procedure_name=procedure_name,
                    steps=[user_input, system_response],
                    success_rate=1.0 if success else 0.0
                )
    
    def _update_semantic_vectors(self):
        """Update semantic search vectors"""
        if not self.semantic_memory:
            return
        
        concepts_text = []
        self.semantic_concepts = []
        
        for memory in self.semantic_memory.values():
            text = f"{memory.concept} {' '.join(memory.relationships)} {str(memory.content)}"
            concepts_text.append(text)
            self.semantic_concepts.append(memory.concept)
        
        if concepts_text:
            self.semantic_vectors = self.vectorizer.fit_transform(concepts_text)
    
    def _persist_semantic_memory(self, memory: SemanticMemory):
        """Persist semantic memory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_memory 
            (id, concept, content, relationships, confidence, importance, access_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id, memory.concept, json.dumps(memory.content),
            json.dumps(memory.relationships), memory.confidence, memory.importance,
            memory.access_count, memory.timestamp, memory.last_accessed
        ))
        
        conn.commit()
        conn.close()
    
    def _persist_episodic_memory(self, memory: EpisodicMemory):
        """Persist episodic memory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO episodic_memory 
            (id, user_id, context, content, outcome, emotional_valence, importance, access_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id, memory.user_id, memory.context, json.dumps(memory.content),
            memory.outcome, memory.emotional_valence, memory.importance,
            memory.access_count, memory.timestamp, memory.last_accessed
        ))
        
        conn.commit()
        conn.close()
    
    def _persist_procedural_memory(self, memory: ProceduralMemory):
        """Persist procedural memory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO procedural_memory 
            (id, procedure_name, steps, success_rate, optimization_level, importance, access_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id, memory.procedure_name, json.dumps(memory.steps),
            memory.success_rate, memory.optimization_level, memory.importance,
            memory.access_count, memory.timestamp, memory.last_accessed
        ))
        
        conn.commit()
        conn.close()
    
    def load_memories(self):
        """Load memories from database"""
        if not os.path.exists(self.db_path):
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load semantic memories
        cursor.execute('SELECT * FROM semantic_memory')
        for row in cursor.fetchall():
            memory = SemanticMemory(
                id=row[0],
                concept=row[1],
                content=json.loads(row[2]),
                relationships=json.loads(row[3]),
                confidence=row[4],
                importance=row[5],
                access_count=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                last_accessed=datetime.fromisoformat(row[8]) if row[8] else None
            )
            self.semantic_memory[memory.id] = memory
        
        # Load procedural memories
        cursor.execute('SELECT * FROM procedural_memory')
        for row in cursor.fetchall():
            memory = ProceduralMemory(
                id=row[0],
                procedure_name=row[1],
                steps=json.loads(row[2]),
                success_rate=row[3],
                optimization_level=row[4],
                importance=row[5],
                access_count=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                last_accessed=datetime.fromisoformat(row[8]) if row[8] else None,
                content={"steps": json.loads(row[2]), "metadata": {}}
            )
            self.procedural_memory[memory.id] = memory
        
        conn.close()
        self._update_semantic_vectors()
    
    def _background_consolidation(self):
        """Background memory consolidation process"""
        while True:
            time.sleep(300)  # Run every 5 minutes
            self._consolidate_memories()
    
    def _consolidate_memories(self):
        """Consolidate and optimize memories"""
        # Implement forgetting curve and memory consolidation
        current_time = datetime.now()
        
        # Consolidate semantic memories
        for memory in list(self.semantic_memory.values()):
            time_diff = (current_time - memory.timestamp).days
            decay_factor = np.exp(-self.forgetting_curve_factor * time_diff)
            memory.importance *= decay_factor
            
            # Remove very low importance memories
            if memory.importance < 0.1 and memory.access_count < 2:
                del self.semantic_memory[memory.id]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "semantic_memories": len(self.semantic_memory),
            "episodic_memories": len(self.episodic_memory),
            "procedural_memories": len(self.procedural_memory),
            "total_memories": len(self.semantic_memory) + len(self.episodic_memory) + len(self.procedural_memory),
            "memory_size_mb": self._calculate_memory_size()
        }
    
    def _calculate_memory_size(self) -> float:
        """Calculate approximate memory size in MB"""
        try:
            if os.path.exists(self.db_path):
                return os.path.getsize(self.db_path) / (1024 * 1024)
        except:
            pass
        return 0.0

# Global memory system instance
memory_system = AdvancedMemorySystem()

# Initialize with basic bioinformatics knowledge
def initialize_bioinformatics_knowledge():
    """Initialize the memory system with basic bioinformatics knowledge"""
    
    # Gene-disease associations
    gene_disease_knowledge = {
        'BRCA1': {
            'diseases': ['breast cancer', 'ovarian cancer'],
            'function': 'DNA repair, tumor suppression',
            'pathways': ['homologous recombination', 'cell cycle checkpoint']
        },
        'BRCA2': {
            'diseases': ['breast cancer', 'ovarian cancer', 'prostate cancer'],
            'function': 'DNA repair, homologous recombination',
            'pathways': ['DNA damage response', 'homologous recombination']
        },
        'TP53': {
            'diseases': ['Li-Fraumeni syndrome', 'various cancers'],
            'function': 'tumor suppressor, cell cycle regulation',
            'pathways': ['p53 signaling', 'apoptosis', 'cell cycle arrest']
        },
        'APOE': {
            'diseases': ['Alzheimer disease', 'cardiovascular disease'],
            'function': 'lipid transport, cholesterol metabolism',
            'pathways': ['lipid metabolism', 'neurodegeneration']
        }
    }
    
    for gene, info in gene_disease_knowledge.items():
        memory_system.store_semantic_memory(
            concept=gene,
            content=info,
            relationships=info['diseases'] + info['pathways'],
            confidence=0.9
        )
    
    # Protein interaction knowledge
    protein_interactions = {
        'BRCA1': ['BRCA2', 'TP53', 'ATM', 'CHEK2', 'RAD51'],
        'BRCA2': ['BRCA1', 'RAD51', 'PALB2'],
        'TP53': ['MDM2', 'ATM', 'CHEK2', 'BRCA1'],
        'ATM': ['BRCA1', 'TP53', 'CHEK2']
    }
    
    for protein, interactions in protein_interactions.items():
        memory_system.store_semantic_memory(
            concept=f"{protein}_interactions",
            content={'protein': protein, 'interactions': interactions},
            relationships=interactions,
            confidence=0.85
        )

# Initialize knowledge when module is imported
initialize_bioinformatics_knowledge()
