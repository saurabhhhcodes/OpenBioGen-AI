"""
Comprehensive Test Script for OpenBioGen AI Advanced System
"""

import os
import sys
from dotenv import load_dotenv
from advanced_main import AdvancedOpenBioGenAI
from advanced_core import AdvancedDataIntegrator, VisualizationEngine, ClinicalDecisionSupport, ReportGenerator
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
    """Test advanced system initialization"""
    print("Testing Advanced System Initialization...")
    
    # Test without API key
    try:
        system = AdvancedOpenBioGenAI()
        print("✅ Advanced system initialized successfully without API key")
    except Exception as e:
        print(f"❌ Advanced system initialization failed: {e}")
        return False
    
    # Test individual components
    try:
        integrator = AdvancedDataIntegrator()
        visualizer = VisualizationEngine()
        clinical = ClinicalDecisionSupport()
        reporter = ReportGenerator()
        print("✅ All advanced components initialized successfully")
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False
    
    return True

def test_advanced_prediction():
    """Test advanced gene-disease prediction functionality"""
    print("Testing Advanced Gene-Disease Prediction...")
    
    try:
        system = AdvancedOpenBioGenAI("demo_key")
        
        # Test single prediction
        result = system.predict_association_comprehensive("BRCA1", "breast cancer", family_history=True)
        assert "gene" in result, "Missing gene in result"
        assert "disease" in result, "Missing disease in result"
        assert "confidence" in result, "Missing confidence in result"
        assert "risk_category" in result, "Missing risk category in result"
        
        # Test batch prediction
        pairs = [("BRCA1", "breast cancer"), ("APOE", "alzheimers disease")]
        batch_results = system.batch_predict_comprehensive(pairs)
        assert len(batch_results) == 2, "Batch prediction failed"
        
        print("✅ Advanced prediction tests passed!")
        return True
    except Exception as e:
        print(f"❌ Advanced prediction test failed: {e}")
        return False

def test_visualization_engine():
    """Test visualization capabilities"""
    print("Testing Visualization Engine...")
    
    try:
        visualizer = VisualizationEngine()
        
        # Test network graph creation
        interactions = {"BRCA1": {"interactions": ["TP53", "ATM", "CHEK2"]}}
        fig = visualizer.create_network_graph("BRCA1", interactions)
        assert fig is not None, "Network graph creation failed"
        
        # Test confidence distribution
        results = [
            {"gene": "BRCA1", "confidence_score": 0.95},
            {"gene": "TP53", "confidence_score": 0.87},
            {"gene": "APOE", "confidence_score": 0.76}
        ]
        fig = visualizer.create_confidence_distribution(results)
        assert fig is not None, "Confidence distribution creation failed"
        
        print("✅ Visualization engine tests passed!")
        return True
    except Exception as e:
        print(f"❌ Visualization engine test failed: {e}")
        return False

def test_clinical_decision_support():
    """Test clinical decision support functionality"""
    print("Testing Clinical Decision Support...")
    
    try:
        clinical = ClinicalDecisionSupport()
        
        # Test risk score calculation
        gene_data = {
            "pathogenic_variants": 100,
            "likely_pathogenic": 50,
            "vus": 200,
            "benign": 300,
            "penetrance": 0.6
        }
        risk_result = clinical.calculate_risk_score(gene_data, family_history=True)
        assert isinstance(risk_result, dict), "Risk result should be a dictionary"
        assert "risk_score" in risk_result, "Missing risk_score in result"
        assert 0 <= risk_result["risk_score"] <= 1, "Risk score out of valid range"
        
        # Test clinical recommendations
        recommendations = clinical.get_clinical_recommendations("BRCA1", "breast cancer", {"risk_score": 0.8})
        assert "recommendations" in recommendations, "Missing recommendations"
        
        print("✅ Clinical decision support tests passed!")
        return True
    except Exception as e:
        print(f"❌ Clinical decision support test failed: {e}")
        return False

def test_report_generation():
    """Test PDF report generation"""
    print("Testing Report Generation...")
    
    try:
        reporter = ReportGenerator()
        
        # Test PDF report generation
        analysis_results = {
            "gene": "BRCA1",
            "disease": "breast cancer",
            "confidence": "High",
            "risk_category": "High",
            "clinical_data": {"pathogenic_variants": 100},
            "literature_summary": "Test summary"
        }
        
        pdf_buffer = reporter.generate_pdf_report("BRCA1", "breast cancer", analysis_results)
        assert pdf_buffer is not None, "PDF generation failed"
        assert len(pdf_buffer.getvalue()) > 0, "PDF buffer is empty"
        
        print("✅ Report generation tests passed!")
        return True
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🧬 OpenBioGen AI - Comprehensive Advanced System Tests")
    print("=" * 60)
    
    load_dotenv()
    
    # Run all tests
    tests = [
        test_gene_validator,
        test_disease_validator,
        test_sample_data,
        test_system_initialization,
        test_advanced_prediction,
        test_visualization_engine,
        test_clinical_decision_support,
        test_report_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All comprehensive tests passed!")
        print("✅ OpenBioGen AI Advanced System is fully functional!")
    else:
        print("⚠️ Some tests failed. Check the output above.")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    main()
