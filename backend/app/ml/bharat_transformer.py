"""
BharatVerify Transformer - Custom transformer architecture for Indian context understanding
Handles multiple Indian languages with community-aware embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional, Tuple

class BharatTokenizer:
    """Custom tokenizer for multiple Indian languages"""
    
    def __init__(self, model_name='ai4bharat/indic-bert'):
        """
        Initialize with IndicBERT tokenizer - supports 12 Indian languages:
        Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, 
        Nepali, Odia, Punjabi, Tamil, Telugu, Urdu + English
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
        
        # Language codes for Indian languages
        self.language_codes = {
            'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'kn': 'Kannada',
            'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'or': 'Odia',
            'pa': 'Punjabi', 'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu', 'en': 'English'
        }
    
    def encode(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs with attention masks"""
        max_len = max_length or self.max_length
        
        encoding = self.tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def detect_language(self, text: str) -> str:
        """Simple language detection for Indian languages"""
        # Simplified language detection based on script/characters
        # In production, use a more sophisticated language detector
        
        # Hindi/Devanagari script
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hi'
        # Bengali script
        elif any('\u0980' <= char <= '\u09FF' for char in text):
            return 'bn'
        # Tamil script
        elif any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'ta'
        # Telugu script
        elif any('\u0C00' <= char <= '\u0C7F' for char in text):
            return 'te'
        # Add more language detection logic as needed
        else:
            return 'en'  # Default to English

class CommunityAwareEmbedding(nn.Module):
    """Embedding layer that combines word embeddings with community context"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_communities: int = 15):
        super().__init__()
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Community embeddings for different Indian communities
        self.community_embedding = nn.Embedding(num_communities, embed_dim)
        
        # Positional embeddings
        self.positional_embedding = nn.Embedding(512, embed_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Community types mapping
        self.community_types = {
            0: 'General', 1: 'Hindu', 2: 'Muslim', 3: 'Christian', 4: 'Sikh',
            5: 'Buddhist', 6: 'Jain', 7: 'LGBTQ+', 8: 'Dalit', 9: 'Tribal',
            10: 'Women', 11: 'Youth', 12: 'Elderly', 13: 'Farmers', 14: 'Regional'
        }
        
    def forward(self, 
                input_ids: torch.Tensor, 
                community_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining word, community, and positional embeddings
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            community_ids: Community type IDs [batch_size] 
            position_ids: Position IDs [batch_size, seq_len]
        
        Returns:
            Combined embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Word embeddings
        word_embeds = self.word_embedding(input_ids)
        
        # Positional embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.positional_embedding(position_ids)
        
        # Combine word and positional embeddings
        embeddings = word_embeds + pos_embeds
        
        # Add community context if provided
        if community_ids is not None:
            community_embeds = self.community_embedding(community_ids).unsqueeze(1)  # [batch_size, 1, embed_dim]
            # Broadcast community embeddings across sequence length
            community_embeds = community_embeds.expand(-1, seq_len, -1)
            embeddings = embeddings + community_embeds
        
        # Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class CulturallyAwareAttention(nn.Module):
    """Multi-head attention with cultural context awareness"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Cultural context projection
        self.cultural_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cultural_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cultural context
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            cultural_context: Cultural context embeddings [batch_size, seq_len, embed_dim]
        
        Returns:
            output: Attended representations [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Add cultural context to keys and values if provided
        if cultural_context is not None:
            cultural_features = self.cultural_proj(cultural_context)
            K = K + cultural_features
            V = V + cultural_features
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attended_values)
        
        return output, attention_weights

class BharatTransformerLayer(nn.Module):
    """Single transformer layer with cultural awareness"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Culturally-aware attention
        self.attention = CulturallyAwareAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cultural_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer layer
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            cultural_context: Cultural context embeddings [batch_size, seq_len, embed_dim]
        
        Returns:
            output: Transformed representations [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attended, attention_weights = self.attention(
            hidden_states, attention_mask, cultural_context
        )
        hidden_states = self.norm1(hidden_states + attended)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states

class BharatVerifyTransformer(nn.Module):
    """Main BharatVerify Transformer model for Indian context understanding"""
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 ff_dim: int = 3072,
                 num_communities: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Embedding layer with community awareness
        self.embeddings = CommunityAwareEmbedding(vocab_size, embed_dim, num_communities)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BharatTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Pooling layer for sequence representation
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                community_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BharatVerify Transformer
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            community_ids: Community type IDs [batch_size]
            position_ids: Position IDs [batch_size, seq_len]
        
        Returns:
            Dictionary containing:
                - last_hidden_state: Final hidden states [batch_size, seq_len, embed_dim]
                - pooler_output: Pooled sequence representation [batch_size, embed_dim]
                - all_hidden_states: Hidden states from all layers (if requested)
        """
        # Get embeddings
        hidden_states = self.embeddings(input_ids, community_ids, position_ids)
        
        # Store all hidden states for analysis
        all_hidden_states = [hidden_states]
        
        # Cultural context (use embeddings as cultural context for now)
        cultural_context = hidden_states
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cultural_context)
            all_hidden_states.append(hidden_states)
        
        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Pooled output (first token representation)
        pooled_output = self.pooler(hidden_states[:, 0])
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output,
            'all_hidden_states': all_hidden_states
        }

# Utility function to create model
def create_bharat_transformer(config: Dict = None) -> BharatVerifyTransformer:
    """Create BharatVerify Transformer with default or custom configuration"""
    
    default_config = {
        'vocab_size': 30000,
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'num_communities': 15,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return BharatVerifyTransformer(**default_config)

# Example usage and testing
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = BharatTokenizer()
    model = create_bharat_transformer()
    
    # Test with sample Indian language text
    test_texts = [
        "प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है।",  # Hindi
        "The Prime Minister said that India's economy is strong.",  # English
        "প্রধানমন্ত্রী বলেছেন যে ভারতের অর্থনীতি শক্তিশালী।"  # Bengali
    ]
    
    for text in test_texts:
        print(f"\nTesting with: {text}")
        
        # Tokenize
        encoded = tokenizer.encode(text)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Detect language and assign community context
        language = tokenizer.detect_language(text)
        community_id = torch.tensor([1])  # Example community ID
        
        print(f"Detected language: {language}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                community_ids=community_id
            )
        
        print(f"Output shape: {outputs['last_hidden_state'].shape}")
        print(f"Pooled output shape: {outputs['pooler_output'].shape}")
        print("✅ Model working successfully!")
