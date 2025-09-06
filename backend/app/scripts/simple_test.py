"""
Simple test script to verify our system works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_domain_classifier():
    """Test the domain classifier"""
    try:
        from app.services.domain_classifier import DomainClassifier, demo_domain_classification
        print("✅ Domain classifier imports successfully!")
        
        # Run demo
        demo_domain_classification()
        return True
        
    except Exception as e:
        print(f"❌ Error testing domain classifier: {e}")
        return False

def test_bharat_transformer():
    """Test the BharatVerify Transformer"""
    try:
        from app.ml.bharat_transformer import BharatTokenizer, create_bharat_transformer
        print("✅ BharatVerify Transformer imports successfully!")
        
        # Create small model for testing
        model = create_bharat_transformer({
            'vocab_size': 1000,
            'embed_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'ff_dim': 512
        })
        
        print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True
        
    except Exception as e:
        print(f"❌ Error testing BharatVerify Transformer: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FACTSY Universal AI Components...")
    print("=" * 50)
    
    # Test components
    domain_success = test_domain_classifier()
    transformer_success = test_bharat_transformer()
    
    if domain_success and transformer_success:
        print("\n🎉 All tests passed! Your system is working!")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
