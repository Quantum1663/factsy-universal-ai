"""
Test suite for BharatVerify Transformer
"""

import torch
import pytest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.bharat_transformer import (
    BharatTokenizer, 
    BharatVerifyTransformer, 
    create_bharat_transformer
)

class TestBharatTransformer:
    """Test cases for BharatVerify Transformer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tokenizer = BharatTokenizer()
        self.model = create_bharat_transformer({
            'vocab_size': 1000,  # Smaller for testing
            'embed_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'ff_dim': 512
        })
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        assert self.tokenizer is not None
        assert hasattr(self.tokenizer, 'tokenizer')
        assert hasattr(self.tokenizer, 'language_codes')
    
    def test_tokenizer_encoding(self):
        """Test text encoding"""
        text = "भारत एक महान देश है"  # Hindi: India is a great country
        encoded = self.tokenizer.encode(text)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[1] == 512  # max_length
        assert encoded['attention_mask'].shape[1] == 512
    
    def test_language_detection(self):
        """Test language detection"""
        hindi_text = "नमस्ते भारत"
        english_text = "Hello India"
        bengali_text = "নমস্কার ভারত"
        
        assert self.tokenizer.detect_language(hindi_text) == 'hi'
        assert self.tokenizer.detect_language(english_text) == 'en'
        assert self.tokenizer.detect_language(bengali_text) == 'bn'
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        community_ids = torch.tensor([1, 2])
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            community_ids=community_ids
        )
        
        # Check outputs
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, 128)
        assert outputs['pooler_output'].shape == (batch_size, 128)
    
    def test_multilingual_processing(self):
        """Test processing of multiple Indian languages"""
        test_cases = [
            ("भारत की अर्थव्यवस्था मजबूत है", 'hi'),  # Hindi
            ("India's economy is strong", 'en'),      # English
            ("ভারতের অর্থনীতি শক্তিশালী", 'bn'),        # Bengali
        ]
        
        for text, expected_lang in test_cases:
            # Tokenize
            encoded = self.tokenizer.encode(text)
            
            # Detect language
            detected_lang = self.tokenizer.detect_language(text)
            assert detected_lang == expected_lang, f"Expected {expected_lang}, got {detected_lang}"
            
            # Process through model
            community_id = torch.tensor([1])
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                community_ids=community_id
            )
            
            # Verify output shapes
            assert outputs['last_hidden_state'].shape[0] == 1  # batch_size
            assert outputs['pooler_output'].shape[0] == 1
            
    def test_community_context_integration(self):
        """Test community context integration"""
        text = "सभी मुसलमान आतंकवादी हैं"  # Problematic claim in Hindi
        encoded = self.tokenizer.encode(text)
        
        # Test with different community contexts
        community_ids = [0, 1, 2]  # Different communities
        
        outputs_list = []
        for comm_id in community_ids:
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                community_ids=torch.tensor([comm_id])
            )
            outputs_list.append(outputs['pooler_output'])
        
        # Outputs should be different for different community contexts
        for i in range(len(outputs_list)):
            for j in range(i+1, len(outputs_list)):
                assert not torch.allclose(outputs_list[i], outputs_list[j], atol=1e-4), \
                    "Community context should produce different representations"

# Run tests
if __name__ == "__main__":
    test_suite = TestBharatTransformer()
    test_suite.setup_method()
    
    print("🧪 Testing BharatVerify Transformer...")
    
    try:
        test_suite.test_tokenizer_initialization()
        print("✅ Tokenizer initialization test passed")
        
        test_suite.test_tokenizer_encoding()
        print("✅ Tokenizer encoding test passed")
        
        test_suite.test_language_detection()
        print("✅ Language detection test passed")
        
        test_suite.test_model_forward_pass()
        print("✅ Model forward pass test passed")
        
        test_suite.test_multilingual_processing()
        print("✅ Multilingual processing test passed")
        
        test_suite.test_community_context_integration()
        print("✅ Community context integration test passed")
        
        print("\n🎉 ALL TESTS PASSED! Your BharatVerify Transformer is working perfectly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise
