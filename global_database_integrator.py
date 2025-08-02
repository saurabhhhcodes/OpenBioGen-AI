"""
Global Database Integrator for OpenBioGen-AI
Integrates PubChem, UniProt, KEGG, Reactome, and other major bioinformatics databases
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from urllib.parse import quote
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging

@dataclass
class DatabaseResult:
    """Standardized result from database queries"""
    source: str
    identifier: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    success: bool = True
    error_message: str = ""

class GlobalDatabaseIntegrator:
    """Comprehensive integration with major bioinformatics databases"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenBioGen-AI/1.0 (Bioinformatics Research Tool)'
        })
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = {
            'pubchem': 0.2,  # 5 requests per second
            'uniprot': 0.1,   # 10 requests per second
            'kegg': 0.5,      # 2 requests per second
            'reactome': 0.3,  # 3 requests per second
            'ensembl': 0.2,   # 5 requests per second
            'ncbi': 0.4       # 2.5 requests per second
        }
    
    def _rate_limit(self, database: str):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        if database in self.last_request_time:
            time_since_last = current_time - self.last_request_time[database]
            min_interval = self.min_request_interval.get(database, 0.5)
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
        self.last_request_time[database] = time.time()
    
    def _make_request(self, url: str, database: str, params: Dict = None, timeout: int = 10) -> Optional[requests.Response]:
        """Make rate-limited HTTP request with error handling"""
        try:
            self._rate_limit(database)
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {database}: {e}")
            return None
    
    # PubChem Integration
    def get_pubchem_compound_data(self, compound_name: str) -> DatabaseResult:
        """Get compound data from PubChem"""
        try:
            # Search for compound by name
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(compound_name)}/JSON"
            response = self._make_request(search_url, 'pubchem')
            
            if not response:
                return DatabaseResult('pubchem', compound_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            data = response.json()
            
            if 'PC_Compounds' not in data:
                return DatabaseResult('pubchem', compound_name, {}, 0.0, datetime.now(), False, "Compound not found")
            
            compound = data['PC_Compounds'][0]
            cid = compound['id']['id']['cid']
            
            # Get detailed properties
            props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChI/JSON"
            props_response = self._make_request(props_url, 'pubchem')
            
            result_data = {
                'cid': cid,
                'molecular_formula': '',
                'molecular_weight': 0,
                'iupac_name': '',
                'canonical_smiles': '',
                'inchi': '',
                'synonyms': []
            }
            
            if props_response:
                props_data = props_response.json()
                if 'PropertyTable' in props_data and 'Properties' in props_data['PropertyTable']:
                    props = props_data['PropertyTable']['Properties'][0]
                    result_data.update({
                        'molecular_formula': props.get('MolecularFormula', ''),
                        'molecular_weight': props.get('MolecularWeight', 0),
                        'iupac_name': props.get('IUPACName', ''),
                        'canonical_smiles': props.get('CanonicalSMILES', ''),
                        'inchi': props.get('InChI', '')
                    })
            
            # Get synonyms
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            syn_response = self._make_request(syn_url, 'pubchem')
            
            if syn_response:
                syn_data = syn_response.json()
                if 'InformationList' in syn_data and 'Information' in syn_data['InformationList']:
                    synonyms = syn_data['InformationList']['Information'][0].get('Synonym', [])
                    result_data['synonyms'] = synonyms[:10]  # Limit to 10 synonyms
            
            return DatabaseResult('pubchem', compound_name, result_data, 0.9, datetime.now())
            
        except Exception as e:
            self.logger.error(f"PubChem error for {compound_name}: {e}")
            return DatabaseResult('pubchem', compound_name, {}, 0.0, datetime.now(), False, str(e))
    
    # UniProt Integration
    def get_uniprot_protein_data(self, protein_name: str) -> DatabaseResult:
        """Get protein data from UniProt"""
        try:
            # Search UniProt
            search_url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                'query': f'gene:{protein_name} OR protein_name:{protein_name}',
                'format': 'json',
                'size': 5
            }
            
            response = self._make_request(search_url, 'uniprot', params)
            
            if not response:
                return DatabaseResult('uniprot', protein_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            data = response.json()
            
            if not data.get('results'):
                return DatabaseResult('uniprot', protein_name, {}, 0.0, datetime.now(), False, "Protein not found")
            
            protein = data['results'][0]
            
            result_data = {
                'accession': protein.get('primaryAccession', ''),
                'protein_name': protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                'gene_names': [gene.get('geneName', {}).get('value', '') for gene in protein.get('genes', [])],
                'organism': protein.get('organism', {}).get('scientificName', ''),
                'function': [],
                'subcellular_location': [],
                'domains': [],
                'sequence_length': protein.get('sequence', {}).get('length', 0),
                'mass': protein.get('sequence', {}).get('molWeight', 0)
            }
            
            # Extract function comments
            for comment in protein.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    for text in comment.get('texts', []):
                        result_data['function'].append(text.get('value', ''))
                elif comment.get('commentType') == 'SUBCELLULAR LOCATION':
                    for location in comment.get('subcellularLocations', []):
                        loc_name = location.get('location', {}).get('value', '')
                        if loc_name:
                            result_data['subcellular_location'].append(loc_name)
            
            # Extract domains
            for feature in protein.get('features', []):
                if feature.get('type') == 'DOMAIN':
                    result_data['domains'].append(feature.get('description', ''))
            
            return DatabaseResult('uniprot', protein_name, result_data, 0.9, datetime.now())
            
        except Exception as e:
            self.logger.error(f"UniProt error for {protein_name}: {e}")
            return DatabaseResult('uniprot', protein_name, {}, 0.0, datetime.now(), False, str(e))
    
    # KEGG Integration
    def get_kegg_pathway_data(self, gene_name: str) -> DatabaseResult:
        """Get pathway data from KEGG"""
        try:
            # Search for gene in KEGG
            search_url = f"http://rest.kegg.jp/find/genes/{gene_name}"
            response = self._make_request(search_url, 'kegg')
            
            if not response:
                return DatabaseResult('kegg', gene_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            search_results = response.text.strip()
            if not search_results:
                return DatabaseResult('kegg', gene_name, {}, 0.0, datetime.now(), False, "Gene not found")
            
            # Get first gene ID
            gene_id = search_results.split('\t')[0].split(':')[1]
            
            # Get gene information
            gene_url = f"http://rest.kegg.jp/get/{gene_id}"
            gene_response = self._make_request(gene_url, 'kegg')
            
            result_data = {
                'gene_id': gene_id,
                'pathways': [],
                'orthologs': [],
                'definition': '',
                'organism': ''
            }
            
            if gene_response:
                gene_info = gene_response.text
                lines = gene_info.split('\n')
                
                current_section = None
                for line in lines:
                    if line.startswith('DEFINITION'):
                        result_data['definition'] = line.split('DEFINITION')[1].strip()
                    elif line.startswith('ORGANISM'):
                        result_data['organism'] = line.split('ORGANISM')[1].strip()
                    elif line.startswith('PATHWAY'):
                        current_section = 'pathway'
                        pathway_info = line.split('PATHWAY')[1].strip()
                        if pathway_info:
                            result_data['pathways'].append(pathway_info)
                    elif line.startswith('ORTHOLOGY'):
                        current_section = 'orthology'
                        ortholog_info = line.split('ORTHOLOGY')[1].strip()
                        if ortholog_info:
                            result_data['orthologs'].append(ortholog_info)
                    elif current_section == 'pathway' and line.startswith('            '):
                        result_data['pathways'].append(line.strip())
                    elif current_section == 'orthology' and line.startswith('            '):
                        result_data['orthologs'].append(line.strip())
            
            return DatabaseResult('kegg', gene_name, result_data, 0.8, datetime.now())
            
        except Exception as e:
            self.logger.error(f"KEGG error for {gene_name}: {e}")
            return DatabaseResult('kegg', gene_name, {}, 0.0, datetime.now(), False, str(e))
    
    # Reactome Integration
    def get_reactome_pathway_data(self, protein_name: str) -> DatabaseResult:
        """Get pathway data from Reactome"""
        try:
            # Search for protein in Reactome
            search_url = f"https://reactome.org/ContentService/search/query"
            params = {
                'query': protein_name,
                'species': 'Homo sapiens',
                'types': 'Pathway'
            }
            
            response = self._make_request(search_url, 'reactome', params)
            
            if not response:
                return DatabaseResult('reactome', protein_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            data = response.json()
            
            if not data.get('results'):
                return DatabaseResult('reactome', protein_name, {}, 0.0, datetime.now(), False, "No pathways found")
            
            result_data = {
                'pathways': [],
                'reactions': [],
                'total_results': data.get('totalCount', 0)
            }
            
            for result in data['results'][:10]:  # Limit to 10 results
                pathway_info = {
                    'id': result.get('stId', ''),
                    'name': result.get('name', ''),
                    'species': result.get('species', [{}])[0].get('displayName', ''),
                    'type': result.get('exactType', ''),
                    'url': f"https://reactome.org/content/detail/{result.get('stId', '')}"
                }
                result_data['pathways'].append(pathway_info)
            
            return DatabaseResult('reactome', protein_name, result_data, 0.8, datetime.now())
            
        except Exception as e:
            self.logger.error(f"Reactome error for {protein_name}: {e}")
            return DatabaseResult('reactome', protein_name, {}, 0.0, datetime.now(), False, str(e))
    
    # Ensembl Integration
    def get_ensembl_gene_data(self, gene_name: str) -> DatabaseResult:
        """Get gene data from Ensembl"""
        try:
            # Search for gene
            search_url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_name}"
            headers = {'Content-Type': 'application/json'}
            
            response = self.session.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 404:
                return DatabaseResult('ensembl', gene_name, {}, 0.0, datetime.now(), False, "Gene not found")
            
            if not response.ok:
                return DatabaseResult('ensembl', gene_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            data = response.json()
            
            result_data = {
                'gene_id': data.get('id', ''),
                'display_name': data.get('display_name', ''),
                'description': data.get('description', ''),
                'biotype': data.get('biotype', ''),
                'chromosome': data.get('seq_region_name', ''),
                'start': data.get('start', 0),
                'end': data.get('end', 0),
                'strand': data.get('strand', 0),
                'assembly': data.get('assembly_name', ''),
                'canonical_transcript': data.get('canonical_transcript', ''),
                'external_db': data.get('source', '')
            }
            
            return DatabaseResult('ensembl', gene_name, result_data, 0.9, datetime.now())
            
        except Exception as e:
            self.logger.error(f"Ensembl error for {gene_name}: {e}")
            return DatabaseResult('ensembl', gene_name, {}, 0.0, datetime.now(), False, str(e))
    
    # NCBI Gene Integration
    def get_ncbi_gene_data(self, gene_name: str) -> DatabaseResult:
        """Get gene data from NCBI Gene database"""
        try:
            # Search NCBI Gene
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'gene',
                'term': f'{gene_name}[Gene Name] AND Homo sapiens[Organism]',
                'retmode': 'json',
                'retmax': 5
            }
            
            response = self._make_request(search_url, 'ncbi', params)
            
            if not response:
                return DatabaseResult('ncbi', gene_name, {}, 0.0, datetime.now(), False, "API request failed")
            
            search_data = response.json()
            
            if not search_data.get('esearchresult', {}).get('idlist'):
                return DatabaseResult('ncbi', gene_name, {}, 0.0, datetime.now(), False, "Gene not found")
            
            gene_id = search_data['esearchresult']['idlist'][0]
            
            # Get gene summary
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'json'
            }
            
            summary_response = self._make_request(summary_url, 'ncbi', params)
            
            result_data = {
                'gene_id': gene_id,
                'symbol': '',
                'description': '',
                'aliases': [],
                'chromosome': '',
                'map_location': '',
                'gene_type': '',
                'organism': 'Homo sapiens'
            }
            
            if summary_response:
                summary_data = summary_response.json()
                if 'result' in summary_data and gene_id in summary_data['result']:
                    gene_info = summary_data['result'][gene_id]
                    result_data.update({
                        'symbol': gene_info.get('name', ''),
                        'description': gene_info.get('description', ''),
                        'aliases': gene_info.get('otheraliases', '').split(', ') if gene_info.get('otheraliases') else [],
                        'chromosome': gene_info.get('chromosome', ''),
                        'map_location': gene_info.get('maplocation', ''),
                        'gene_type': gene_info.get('geneticsource', '')
                    })
            
            return DatabaseResult('ncbi', gene_name, result_data, 0.9, datetime.now())
            
        except Exception as e:
            self.logger.error(f"NCBI error for {gene_name}: {e}")
            return DatabaseResult('ncbi', gene_name, {}, 0.0, datetime.now(), False, str(e))
    
    # Comprehensive Search
    def comprehensive_search(self, query: str, search_type: str = 'auto') -> Dict[str, DatabaseResult]:
        """Perform comprehensive search across all databases"""
        results = {}
        
        if search_type in ['auto', 'protein', 'gene']:
            # UniProt search
            results['uniprot'] = self.get_uniprot_protein_data(query)
            
            # KEGG search
            results['kegg'] = self.get_kegg_pathway_data(query)
            
            # Reactome search
            results['reactome'] = self.get_reactome_pathway_data(query)
            
            # Ensembl search
            results['ensembl'] = self.get_ensembl_gene_data(query)
            
            # NCBI Gene search
            results['ncbi'] = self.get_ncbi_gene_data(query)
        
        if search_type in ['auto', 'compound', 'drug']:
            # PubChem search
            results['pubchem'] = self.get_pubchem_compound_data(query)
        
        return results
    
    def get_integrated_summary(self, query: str) -> Dict[str, Any]:
        """Get integrated summary from all relevant databases"""
        results = self.comprehensive_search(query)
        
        summary = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'databases_searched': len(results),
            'successful_results': sum(1 for r in results.values() if r.success),
            'data_sources': [],
            'integrated_data': {
                'identifiers': {},
                'descriptions': [],
                'pathways': [],
                'interactions': [],
                'functions': [],
                'locations': [],
                'chemical_properties': {}
            }
        }
        
        for db_name, result in results.items():
            if result.success:
                summary['data_sources'].append(db_name)
                
                # Integrate identifiers
                if db_name == 'uniprot' and 'accession' in result.data:
                    summary['integrated_data']['identifiers']['uniprot'] = result.data['accession']
                elif db_name == 'ensembl' and 'gene_id' in result.data:
                    summary['integrated_data']['identifiers']['ensembl'] = result.data['gene_id']
                elif db_name == 'ncbi' and 'gene_id' in result.data:
                    summary['integrated_data']['identifiers']['ncbi'] = result.data['gene_id']
                elif db_name == 'pubchem' and 'cid' in result.data:
                    summary['integrated_data']['identifiers']['pubchem'] = result.data['cid']
                
                # Integrate descriptions
                if 'description' in result.data and result.data['description']:
                    summary['integrated_data']['descriptions'].append({
                        'source': db_name,
                        'description': result.data['description']
                    })
                
                # Integrate pathways
                if 'pathways' in result.data:
                    for pathway in result.data['pathways']:
                        if isinstance(pathway, dict):
                            summary['integrated_data']['pathways'].append({
                                'source': db_name,
                                'name': pathway.get('name', ''),
                                'id': pathway.get('id', '')
                            })
                        else:
                            summary['integrated_data']['pathways'].append({
                                'source': db_name,
                                'name': str(pathway),
                                'id': ''
                            })
                
                # Integrate functions
                if 'function' in result.data:
                    for func in result.data['function']:
                        summary['integrated_data']['functions'].append({
                            'source': db_name,
                            'function': func
                        })
                
                # Integrate chemical properties
                if db_name == 'pubchem':
                    summary['integrated_data']['chemical_properties'] = {
                        'molecular_formula': result.data.get('molecular_formula', ''),
                        'molecular_weight': result.data.get('molecular_weight', 0),
                        'smiles': result.data.get('canonical_smiles', ''),
                        'iupac_name': result.data.get('iupac_name', '')
                    }
        
        return summary

# Global instance
global_db_integrator = GlobalDatabaseIntegrator()
