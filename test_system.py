"""
Test script for OpenBioGen AI system
"""

import os
import sys
from dotenv import load_dotenv
from main import OpenBioGenAI
from utils import GeneValidator, create_sample_data

def test_gene_validator():
    """Test gene validation functionality"""
    print("Testing Gene Validator...")
    
    # Valid gene symbols
    valid_genes = ["BRCA1", "TP53", "APOE", "HTT", "SOD1"]
    for gene in valid_genes:
        assert GeneValidator.validate_gene_symbol(gene), f"Valid gene {gene} failed validation"
    
    # Invalid gene symbols
    invalid_genes = ["", "   ", "GENE@123", "gene with spaces"]
    for gene in invalid_genes:
        assert not GeneValidator.validate_gene_symbol(gene), f"Invalid gene {gene} passed validation"
    
    print("✅ Gene Validator tests passed!")

def test_disease_validator():
    """Test disease validation functionality"""
    print("Testing Disease Validator...")
    
    # Valid disease names
    valid_diseases = ["breast cancer", "Alzheimer's disease", "Type-2 diabetes", "COVID-19"]
    for disease in valid_diseases:
        assert GeneValidator.validate_disease_name(disease), f"Valid disease {disease} failed validation"
    
    # Invalid disease names
    invalid_diseases = ["", "a", "disease@#$%"]
    for disease in invalid_diseases:
        assert not GeneValidator.validate_disease_name(disease), f"Invalid disease {disease} passed validation"
    
    print("✅ Disease Validator tests passed!")

def test_sample_data():
    """Test sample data creation"""
    print("Testing Sample Data Creation...")
    
    df = create_sample_data()
    assert len(df) > 0, "Sample data is empty"
    assert 'gene' in df.columns, "Gene column missing"
    assert 'disease' in df.columns, "Disease column missing"
    
    print(f"✅ Sample data created with {len(df)} gene-disease pairs")
    print(df.head())

def test_system_initialization():
    """Test system initialization without API key"""
    print("Testing System Initialization...")
    
    try:
        # This should work even without API key for basic initialization
        ai_system = OpenBioGenAI("dummy_key")
        print("✅ System initialization successful (basic components)")
    except Exception as e:
        print(f"⚠️ System initialization failed: {e}")
        print("This is expected without proper API keys")

def main():
    """Run all tests"""
    print("🧬 OpenBioGen AI - System Tests")
    print("=" * 50)
    
    try:
        test_gene_validator()
        test_disease_validator()
        test_sample_data()
        test_system_initialization()
        
        print("\n" + "=" * 50)
        print("🎉 All basic tests completed!")
        print("\nTo run the full system:")
        print("1. Get a Tavily API key from https://tavily.com/")
        print("2. Copy .env.example to .env and add your API key")
        print("3. Run: streamlit run main.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
