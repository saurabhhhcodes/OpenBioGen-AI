"""
OpenBioGen AI - Advanced Core System
Professional Bioinformatics Platform with Comprehensive Analytics
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import requests
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

class AdvancedDataIntegrator:
    """Advanced data integration with multiple bioinformatics databases"""
    
    def __init__(self):
        # Enhanced mock databases with realistic data
        self.clinvar_data = {
            "BRCA1": {"pathogenic_variants": 1847, "likely_pathogenic": 423, "vus": 2156, "benign": 891},
            "TP53": {"pathogenic_variants": 2341, "likely_pathogenic": 567, "vus": 3421, "benign": 1234},
            "APOE": {"pathogenic_variants": 234, "likely_pathogenic": 89, "vus": 456, "benign": 678}
        }
        
        self.gwas_data = {
            "BRCA1": {
                "breast_cancer": {"p_value": 5.2e-45, "odds_ratio": 12.3, "sample_size": 122000},
                "ovarian_cancer": {"p_value": 2.1e-38, "odds_ratio": 8.7, "sample_size": 89000}
            },
            "APOE": {"alzheimers_disease": {"p_value": 1.3e-89, "odds_ratio": 3.2, "sample_size": 455000}},
            "TP53": {"lung_cancer": {"p_value": 4.7e-23, "odds_ratio": 2.1, "sample_size": 234000}}
        }
        
        self.string_data = {
            "BRCA1": {
                "interacting_proteins": ["BRCA2", "ATM", "CHEK2", "TP53", "RAD51"],
                "confidence_scores": [0.999, 0.995, 0.987, 0.923, 0.876],
                "pathways": ["DNA repair", "Cell cycle checkpoint", "Homologous recombination"]
            },
            "TP53": {
                "interacting_proteins": ["MDM2", "ATM", "CHEK2", "BRCA1", "RB1"],
                "confidence_scores": [0.999, 0.998, 0.989, 0.923, 0.845],
                "pathways": ["p53 signaling", "Cell cycle", "Apoptosis", "DNA damage response"]
            },
            "INS": {
                "interacting_proteins": ["INSR", "IGF1R", "IRS1", "IRS2", "PIK3CA", "AKT1", "FOXO1"],
                "confidence_scores": [0.999, 0.987, 0.976, 0.965, 0.943, 0.932, 0.876],
                "pathways": ["Insulin signaling", "Glucose metabolism", "PI3K-Akt signaling", "FOXO signaling"]
            },
            "INSULIN": {
                "interacting_proteins": ["INSR", "IGF1R", "IRS1", "IRS2", "PIK3CA", "AKT1", "FOXO1"],
                "confidence_scores": [0.999, 0.987, 0.976, 0.965, 0.943, 0.932, 0.876],
                "pathways": ["Insulin signaling", "Glucose metabolism", "PI3K-Akt signaling", "FOXO signaling"]
            },
            "INSR": {
                "interacting_proteins": ["INS", "IGF1", "IRS1", "IRS2", "SHC1", "GRB2"],
                "confidence_scores": [0.999, 0.876, 0.965, 0.954, 0.832, 0.821],
                "pathways": ["Insulin receptor signaling", "Growth factor signaling", "MAPK signaling"]
            },
            "APOE": {
                "interacting_proteins": ["LDLR", "LRP1", "ABCA1", "APOA1", "LCAT"],
                "confidence_scores": [0.987, 0.965, 0.943, 0.921, 0.876],
                "pathways": ["Lipid metabolism", "Cholesterol transport", "Alzheimer disease pathway"]
            },
            "HEMOGLOBIN": {
                "interacting_proteins": ["HBA1", "HBA2", "HBB", "HBG1", "HBG2"],
                "confidence_scores": [0.999, 0.999, 0.999, 0.876, 0.865],
                "pathways": ["Oxygen transport", "Heme biosynthesis", "Iron metabolism"]
            },
            "HBA1": {
                "interacting_proteins": ["HBB", "HBA2", "AHSP", "HBG1"],
                "confidence_scores": [0.999, 0.987, 0.876, 0.754],
                "pathways": ["Oxygen transport", "Hemoglobin assembly", "Erythropoiesis"]
            },
            "HBB": {
                "interacting_proteins": ["HBA1", "HBA2", "HBG1", "HBG2", "HBD"],
                "confidence_scores": [0.999, 0.987, 0.876, 0.865, 0.754],
                "pathways": ["Oxygen transport", "Hemoglobin assembly", "Sickle cell disease"]
            },
            "ALBUMIN": {
                "interacting_proteins": ["ALB", "FGA", "FGB", "FGG", "SERPINA1"],
                "confidence_scores": [0.999, 0.765, 0.754, 0.743, 0.732],
                "pathways": ["Protein transport", "Oncotic pressure regulation", "Drug binding"]
            },
            "ALB": {
                "interacting_proteins": ["FGA", "FGB", "FGG", "SERPINA1", "APOA1"],
                "confidence_scores": [0.765, 0.754, 0.743, 0.732, 0.721],
                "pathways": ["Protein transport", "Oncotic pressure regulation", "Lipid transport"]
            },
            "MYOSIN": {
                "interacting_proteins": ["MYH1", "MYH2", "MYH7", "ACTA1", "ACTC1", "TPM1"],
                "confidence_scores": [0.987, 0.976, 0.965, 0.954, 0.943, 0.876],
                "pathways": ["Muscle contraction", "Cytoskeleton organization", "Cardiac muscle contraction"]
            },
            "ACTIN": {
                "interacting_proteins": ["ACTA1", "ACTB", "ACTC1", "MYH1", "TPM1", "TMOD1"],
                "confidence_scores": [0.999, 0.987, 0.976, 0.954, 0.876, 0.765],
                "pathways": ["Cytoskeleton organization", "Cell motility", "Muscle contraction"]
            },
            "COLLAGEN": {
                "interacting_proteins": ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1"],
                "confidence_scores": [0.999, 0.987, 0.876, 0.765, 0.754],
                "pathways": ["Extracellular matrix organization", "Collagen biosynthesis", "Wound healing"]
            },
            "KERATIN": {
                "interacting_proteins": ["KRT1", "KRT10", "KRT14", "KRT5", "KRT15"],
                "confidence_scores": [0.987, 0.976, 0.965, 0.954, 0.876],
                "pathways": ["Intermediate filament organization", "Epithelial cell differentiation", "Skin barrier function"]
            },
            "TUBULIN": {
                "interacting_proteins": ["TUBA1A", "TUBB", "TUBB3", "TUBB4B", "MAP2"],
                "confidence_scores": [0.999, 0.987, 0.876, 0.765, 0.754],
                "pathways": ["Microtubule organization", "Cell division", "Intracellular transport"]
            }
        }
    
    def get_clinvar_data(self, gene: str) -> Dict[str, Any]:
        return self.clinvar_data.get(gene.upper(), {"pathogenic_variants": 0, "likely_pathogenic": 0, "vus": 0, "benign": 0})
    
    def get_gwas_data(self, gene: str, disease: str) -> Dict[str, Any]:
        gene_data = self.gwas_data.get(gene.upper(), {})
        disease_key = disease.lower().replace(" ", "_").replace("'", "")
        return gene_data.get(disease_key, {})
    
    def get_string_data(self, gene: str) -> Dict[str, Any]:
        """Get protein interactions with enhanced fallback and fuzzy matching"""
        gene_upper = gene.upper()
        
        # Direct match
        if gene_upper in self.string_data:
            return self.string_data[gene_upper]
        
        # Fuzzy matching for partial matches
        fuzzy_matches = []
        for known_protein in self.string_data.keys():
            if gene_upper in known_protein or known_protein in gene_upper:
                fuzzy_matches.append(known_protein)
            elif len(gene_upper) > 3:
                # Check for prefix matches
                if gene_upper.startswith(known_protein[:3]) or known_protein.startswith(gene_upper[:3]):
                    fuzzy_matches.append(known_protein)
        
        if fuzzy_matches:
            # Use the best fuzzy match
            best_match = fuzzy_matches[0]
            data = self.string_data[best_match].copy()
            # Reduce confidence for fuzzy matches
            data['confidence_scores'] = [score * 0.8 for score in data['confidence_scores']]
            return data
        
        # Generate synthetic interactions for unknown proteins
        return self._generate_synthetic_interactions(gene)
    
    def _generate_synthetic_interactions(self, protein_name):
        """Generate synthetic interactions for unknown proteins"""
        import random
        
        # Common protein families and their typical interactions
        protein_families = {
            'kinase': {
                'interactions': ['ATP', 'ADP', 'substrate_proteins', 'regulatory_kinases'],
                'pathways': ['Protein phosphorylation', 'Signal transduction', 'Cell cycle regulation']
            },
            'receptor': {
                'interactions': ['ligands', 'signaling_proteins', 'adaptor_proteins', 'kinases'],
                'pathways': ['Signal transduction', 'Cell communication', 'Membrane signaling']
            },
            'enzyme': {
                'interactions': ['substrates', 'cofactors', 'regulatory_proteins', 'ATP'],
                'pathways': ['Metabolic processes', 'Enzymatic reactions', 'Cellular metabolism']
            },
            'transcription': {
                'interactions': ['DNA', 'RNA_polymerase', 'transcription_factors', 'chromatin_proteins'],
                'pathways': ['Gene expression', 'Transcriptional regulation', 'Chromatin remodeling']
            },
            'transport': {
                'interactions': ['membrane_proteins', 'ATP', 'ion_channels', 'carrier_proteins'],
                'pathways': ['Membrane transport', 'Ion transport', 'Cellular transport']
            }
        }
        
        # Default interactions for any protein
        default_interactions = ['ATP', 'ADP', 'GTP', 'GDP', 'water', 'ions']
        default_pathways = ['General metabolism', 'Cellular processes', 'Protein interactions']
        
        # Determine protein family based on name patterns
        protein_lower = protein_name.lower()
        selected_family = None
        
        for family, data in protein_families.items():
            family_keywords = {
                'kinase': ['kinase', 'phos', 'kin'],
                'receptor': ['receptor', 'recep', 'r_', '_r'],
                'enzyme': ['ase', 'enzyme', 'enz'],
                'transcription': ['transcr', 'tf_', '_tf', 'factor'],
                'transport': ['transport', 'channel', 'pump', 'carrier']
            }
            
            if any(keyword in protein_lower for keyword in family_keywords.get(family, [])):
                selected_family = family
                break
        
        # Generate interactions
        if selected_family:
            family_data = protein_families[selected_family]
            interactions = family_data['interactions'][:4] + default_interactions[:3]
            pathways = family_data['pathways']
        else:
            interactions = default_interactions[:5]
            pathways = default_pathways
        
        # Generate confidence scores
        confidence_scores = [0.6 - (i * 0.05) for i in range(len(interactions))]
        
        return {
            "interacting_proteins": interactions,
            "confidence_scores": confidence_scores,
            "pathways": pathways,
            "note": "Predicted interactions (synthetic)"
        }
    
    def integrate_multi_source_data(self, gene: str, disease: str) -> Dict[str, Any]:
        """Integrate data from multiple bioinformatics sources"""
        clinvar_data = self.get_clinvar_data(gene)
        gwas_data = self.get_gwas_data(gene, disease)
        string_data = self.get_string_data(gene)
        
        # Calculate penetrance based on variant data
        total_variants = sum(clinvar_data.values())
        pathogenic_ratio = (clinvar_data.get('pathogenic_variants', 0) + 
                           clinvar_data.get('likely_pathogenic', 0)) / max(total_variants, 1)
        
        integrated_data = {
            'clinical_data': {
                **clinvar_data,
                'penetrance': min(pathogenic_ratio * 0.8, 0.95),  # Cap at 95%
                'lifetime_risk': f"{pathogenic_ratio * 100:.1f}%",
                'clinical_actionability': 'High' if pathogenic_ratio > 0.1 else 'Medium' if pathogenic_ratio > 0.05 else 'Low'
            },
            'gwas_data': gwas_data,
            'protein_interactions': string_data,
            'sources': ['ClinVar', 'GWAS Catalog', 'STRING'],
            'integration_timestamp': datetime.now().isoformat()
        }
        
        return integrated_data

class VisualizationEngine:
    """Advanced visualization capabilities"""
    
    @staticmethod
    def create_network_graph(gene: str, interactions: Dict[str, Any]) -> go.Figure:
        """Create gene-protein interaction network"""
        if not interactions:
            return go.Figure()
        
        G = nx.Graph()
        G.add_node(gene, node_type='query', size=30)
        
        proteins = interactions.get('interacting_proteins', [])
        scores = interactions.get('confidence_scores', [])
        
        for i, protein in enumerate(proteins[:8]):
            score = scores[i] if i < len(scores) else 0.5
            G.add_node(protein, node_type='interaction', size=20)
            G.add_edge(gene, protein, weight=score)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            if G.nodes[node]['node_type'] == 'query':
                node_color.append('red')
                node_size.append(30)
            else:
                node_color.append('lightblue')
                node_size.append(20)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='gray'), 
                                hoverinfo='none', mode='lines', showlegend=False))
        
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
                                text=node_text, textposition="middle center",
                                marker=dict(size=node_size, color=node_color, line=dict(width=2, color='black')),
                                showlegend=False))
        
        fig.update_layout(title=f"Protein Interaction Network for {gene}", showlegend=False,
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        return fig
    
    @staticmethod
    def create_confidence_distribution(results: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence score distribution"""
        confidence_levels = [r.get('confidence', 'Unknown') for r in results]
        confidence_counts = pd.Series(confidence_levels).value_counts()
        
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red', 'Unknown': 'gray'}
        
        fig = go.Figure(data=[go.Bar(x=confidence_counts.index, y=confidence_counts.values,
                                    marker_color=[colors.get(level, 'gray') for level in confidence_counts.index])])
        
        fig.update_layout(title="Confidence Level Distribution", xaxis_title="Confidence Level",
                         yaxis_title="Number of Associations", showlegend=False)
        
        return fig

class ClinicalDecisionSupport:
    """Clinical decision support tools"""
    
    @staticmethod
    def calculate_risk_score(gene_data: Dict[str, Any], family_history: bool = False) -> Dict[str, Any]:
        """Calculate comprehensive risk score"""
        base_penetrance = gene_data.get('penetrance', 0.1)
        
        if family_history:
            adjusted_penetrance = min(base_penetrance * 1.5, 1.0)
        else:
            adjusted_penetrance = base_penetrance
        
        if adjusted_penetrance >= 0.8:
            risk_category, color = "Very High", "red"
        elif adjusted_penetrance >= 0.5:
            risk_category, color = "High", "orange"
        elif adjusted_penetrance >= 0.2:
            risk_category, color = "Moderate", "yellow"
        else:
            risk_category, color = "Low", "green"
        
        return {
            "risk_score": adjusted_penetrance,
            "risk_category": risk_category,
            "color": color,
            "lifetime_risk": gene_data.get('lifetime_risk', 'Unknown'),
            "clinical_actionability": gene_data.get('clinical_actionability', 'Unknown')
        }
    
    @staticmethod
    def get_clinical_recommendations(gene: str, disease: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical recommendations"""
        recommendations = []
        risk_category = risk_data.get('risk_category', 'Unknown')
        
        if risk_category in ['Very High', 'High']:
            recommendations.extend([
                "Consider genetic counseling for patient and family members",
                "Discuss enhanced screening protocols with oncology team",
                "Evaluate for prophylactic interventions based on guidelines",
                "Consider cascade testing for at-risk family members"
            ])
        elif risk_category == 'Moderate':
            recommendations.extend([
                "Consider genetic counseling consultation",
                "Discuss modified screening recommendations",
                "Monitor for additional risk factors"
            ])
        else:
            recommendations.extend([
                "Standard population screening recommendations apply",
                "Consider family history and other risk factors"
            ])
        
        if gene.upper() == 'BRCA1':
            recommendations.extend([
                "Consider MRI screening starting at age 25-30",
                "Discuss prophylactic mastectomy and oophorectomy options",
                "Annual mammography and clinical breast exams"
            ])
        elif gene.upper() == 'TP53':
            recommendations.extend([
                "Comprehensive cancer surveillance protocol",
                "Avoid radiation exposure when possible",
                "Consider whole-body MRI screening"
            ])
        
        return {
            "recommendations": recommendations,
            "risk_category": risk_category,
            "gene_specific": True if gene.upper() in ['BRCA1', 'TP53', 'APOE'] else False
        }

class ReportGenerator:
    """Professional report generation"""
    
    @staticmethod
    def generate_pdf_report(gene: str, disease: str, analysis_results: Dict[str, Any]) -> bytes:
        """Generate comprehensive PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        pdf.cell(0, 10, f'OpenBioGen AI Analysis Report', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Gene: {gene.upper()} | Disease: {disease.title()}', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis Summary', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        confidence = analysis_results.get('confidence', 'Unknown')
        pdf.cell(0, 8, f'Confidence Level: {confidence}', 0, 1)
        
        risk_category = analysis_results.get('risk_category', 'Unknown')
        pdf.cell(0, 8, f'Risk Category: {risk_category}', 0, 1)
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Clinical Prediction', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        prediction = analysis_results.get('prediction', 'No prediction available')
        words = prediction.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 80:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            pdf.cell(0, 6, line, 0, 1)
        
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Clinical Recommendations', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for i, rec in enumerate(recommendations[:8], 1):
                pdf.cell(0, 6, f'{i}. {rec}', 0, 1)
        
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f'Generated by OpenBioGen AI on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        
        from io import BytesIO
        buffer = BytesIO()
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            buffer.write(pdf_output.encode('latin-1'))
        else:
            buffer.write(pdf_output)
        buffer.seek(0)
        return buffer

class AdvancedOpenBioGenAI:
    """Main AI system integrating all components"""
    
    def __init__(self, api_key: str = None):
        self.data_integrator = AdvancedDataIntegrator()
        self.visualizer = VisualizationEngine()
        self.clinical_support = ClinicalDecisionSupport()
        self.report_generator = ReportGenerator()
        
        # Initialize Tavily client if API key is provided
        self.api_key = api_key
        if api_key and TAVILY_AVAILABLE:
            try:
                self.retriever = AdvancedTavilyRetriever(api_key)
                self.use_real_api = True
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily client: {e}")
                self.retriever = AdvancedTavilyRetriever()
                self.use_real_api = False
        else:
            self.retriever = AdvancedTavilyRetriever()
            self.use_real_api = False
    
    def predict_association_comprehensive(self, gene: str, disease: str) -> Dict[str, Any]:
        """Comprehensive gene-disease association prediction"""
        try:
            # Get integrated data
            integrated_data = self.data_integrator.integrate_multi_source_data(gene, disease)
            
            # Calculate risk score
            risk_data = self.clinical_support.calculate_risk_score(integrated_data)
            
            # Get clinical recommendations
            recommendations = self.clinical_support.get_clinical_recommendations(gene, disease, risk_data)
            
            # Get literature search results
            literature_results = self.retriever.search_literature(f"{gene} {disease} association")
            
            # Combine all results
            comprehensive_result = {
                'gene': gene,
                'disease': disease,
                'integrated_data': integrated_data,
                'risk_assessment': risk_data,
                'clinical_recommendations': recommendations,
                'literature_evidence': literature_results,
                'confidence': self._calculate_overall_confidence(integrated_data, literature_results),
                'timestamp': datetime.now().isoformat()
            }
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {e}")
            return {
                'gene': gene,
                'disease': disease,
                'error': str(e),
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_analysis(self, gene_disease_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Perform batch analysis on multiple gene-disease pairs"""
        results = []
        
        for gene, disease in gene_disease_pairs:
            try:
                result = self.predict_association_comprehensive(gene, disease)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {gene}-{disease}: {e}")
                results.append({
                    'gene': gene,
                    'disease': disease,
                    'error': str(e),
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _calculate_overall_confidence(self, integrated_data: Dict[str, Any], literature_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from clinical data
            clinical_data = integrated_data.get('clinical_data', {})
            penetrance = clinical_data.get('penetrance', 0.0)
            
            # Literature confidence
            lit_confidence = len(literature_results) / 10.0  # Normalize to 0-1
            lit_confidence = min(lit_confidence, 1.0)
            
            # Combined confidence
            overall_confidence = (penetrance * 0.7) + (lit_confidence * 0.3)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence

class AdvancedTavilyRetriever:
    """Advanced retriever with real and mock API integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.use_real_api = False
        
        if api_key and TAVILY_AVAILABLE:
            try:
                self.client = TavilyClient(api_key=api_key)
                self.use_real_api = True
                logger.info("Using real Tavily API with advanced features")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily client: {e}")
                self.use_real_api = False
        
        if not self.use_real_api:
            logger.info("Using mock Tavily API for demonstration")
    
    def search_literature(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search literature with real or mock API"""
        if self.use_real_api:
            return self._search_real_api(query, max_results)
        else:
            return self._search_mock_api(query, max_results)
    
    def _search_real_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using real Tavily API"""
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org"]
            )
            
            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.0),
                    'source': 'Tavily API'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error with real Tavily API: {e}")
            return self._search_mock_api(query, max_results)
    
    def _search_mock_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Mock literature search for demonstration"""
        mock_results = [
            {
                'title': f'Comprehensive analysis of {query.split()[0]} in disease pathogenesis',
                'url': 'https://pubmed.ncbi.nlm.nih.gov/mock1',
                'content': f'This study investigates the role of {query.split()[0]} in disease development...',
                'score': 0.95,
                'source': 'Mock API'
            },
            {
                'title': f'Clinical implications of {query.split()[0]} variants',
                'url': 'https://nature.com/mock2',
                'content': f'Recent findings suggest that {query.split()[0]} variants have significant clinical impact...',
                'score': 0.87,
                'source': 'Mock API'
            },
            {
                'title': f'Therapeutic targeting of {query.split()[0]} pathway',
                'url': 'https://science.org/mock3',
                'content': f'Novel therapeutic approaches targeting {query.split()[0]} show promising results...',
                'score': 0.82,
                'source': 'Mock API'
            }
        ]
        
        return mock_results[:max_results]
