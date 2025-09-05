"""
FACTSY Universal AI - Machine Learning Module

This module contains the revolutionary BharatVerify AI models:
- BharatVerify Transformer: Custom transformer for Indian context understanding
- Community Context Engine: Community-aware misinformation detection
- Domain-specific classifiers: Politics, Health, Economics, etc.
- Multi-modal analysis: Text, images, audio, video processing
"""

from .bharat_transformer import (
    BharatTokenizer,
    CommunityAwareEmbedding,
    CulturallyAwareAttention,
    BharatVerifyTransformer,
    create_bharat_transformer
)

__all__ = [
    'BharatTokenizer',
    'CommunityAwareEmbedding', 
    'CulturallyAwareAttention',
    'BharatVerifyTransformer',
    'create_bharat_transformer'
]
