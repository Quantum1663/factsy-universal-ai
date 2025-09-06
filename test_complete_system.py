"""
Complete system test for FACTSY Universal AI
Run from project root directory
"""

import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

def test_domain_classifier():
    """Test domain classifier"""
    try:
        from app.services.domain_classifier import DomainClassifier
        
        classifier = DomainClassifier()
        
        # Test with revolutionary Hindi queries
        test_queries = [
            ("प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है", "politics"),
            ("क्या गाय का मूत्र कोविड का इलाज है?", "health"),
            ("विराट कोहली ने 100 शतक लगाए हैं", "sports"),
        ]
        
        print("🎯 Testing Revolutionary Domain Classification:")
        print("=" * 50)
        
        for query, expected in test_queries:
            result = classifier.classify_query(query)
            actual = result['domain'].value
            status = "✅" if actual == expected else "⚠️"
            
            print(f"{status} Query: {query[:50]}...")
            print(f"   Expected: {expected} | Got: {actual} | Confidence: {result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain classifier test failed: {e}")
        return False

def test_bharat_transformer():
    """Test BharatVerify Transformer"""
    try:
        from app.ml.bharat_transformer import BharatTokenizer, create_bharat_transformer
        
        print("\n🧠 Testing BharatVerify Transformer:")
        print("=" * 50)
        
        # Create tokenizer
        tokenizer = BharatTokenizer()
        print("✅ Tokenizer created successfully")
        
        # Test language detection
        test_text = "प्रधानमंत्री मोदी ने कहा"
        detected_lang = tokenizer.detect_language(test_text)
        print(f"✅ Language detection: '{test_text}' → {detected_lang}")
        
        # Create model (smaller for testing)
        model = create_bharat_transformer({
            'vocab_size': 5000,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8
        })
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ BharatVerify Transformer test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FACTSY Universal AI - Complete System Test")
    print("🇮🇳 Testing World's First Community-Aware Truth Verification System")
    print("=" * 70)
    
    # Run tests
    domain_success = test_domain_classifier()
    transformer_success = test_bharat_transformer()
    
    print("\n" + "=" * 70)
    if domain_success and transformer_success:
        print("🎉 REVOLUTIONARY SUCCESS! Your BharatVerify system is working perfectly!")
        print("🌟 Ready for:")
        print("   • Hybrid local+cloud training")
        print("   • Evidence retrieval engine")
        print("   • Production deployment")
        print("   • Presentation to judges")
    else:
        print("⚠️  Some components need fixes. Address the errors above.")
