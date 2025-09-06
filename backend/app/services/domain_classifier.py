"""
Domain Classification Module for BharatVerify Transformer
Routes queries to appropriate domain-specific engines
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class Domain(Enum):
    """Supported domains for Indian fact-checking"""
    POLITICS = "politics"
    HEALTH = "health" 
    ECONOMICS = "economics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    COMMUNITY = "community"
    CULTURE = "culture"
    TECHNOLOGY = "technology"
    GENERAL = "general"

class DomainClassifier(nn.Module):
    """Classifies queries into appropriate domains for specialized processing"""
    
    def __init__(self, input_dim: int = 768, num_domains: int = 9):
        super().__init__()
        
        self.domains = list(Domain)
        self.domain_keywords = {
            Domain.POLITICS: [
                'मोदी', 'modi', 'राहुल', 'rahul', 'गांधी', 'gandhi', 'भाजपा', 'bjp', 
                'कांग्रेस', 'congress', 'चुनाव', 'election', 'सरकार', 'government', 
                'नीति', 'policy', 'संसद', 'parliament', 'मंत्री', 'minister'
            ],
            Domain.HEALTH: [
                'कोविड', 'covid', 'वैक्सीन', 'vaccine', 'दवा', 'medicine', 'डॉक्टर', 'doctor',
                'आयुर्वेद', 'ayurveda', 'होम्योपैथी', 'homeopathy', 'इलाज', 'treatment',
                'बीमारी', 'disease', 'स्वास्थ्य', 'health', 'अस्पताल', 'hospital'
            ],
            Domain.ECONOMICS: [
                'जीडीपी', 'gdp', 'महंगाई', 'inflation', 'अर्थव्यवस्था', 'economy',
                'रुपया', 'rupee', 'बैंक', 'bank', 'आरबीआई', 'rbi', 'बजट', 'budget',
                'कर', 'tax', 'नौकरी', 'employment', 'बेरोजगारी', 'unemployment'
            ],
            Domain.SPORTS: [
                'विराट', 'virat', 'कोहली', 'kohli', 'धोनी', 'dhoni', 'क्रिकेट', 'cricket',
                'आईपीएल', 'ipl', 'फुटबॉल', 'football', 'हॉकी', 'hockey', 'ओलंपिक', 'olympics',
                'खेल', 'sports', 'मैच', 'match', 'स्कोर', 'score'
            ],
            Domain.ENTERTAINMENT: [
                'बॉलीवुड', 'bollywood', 'शाहरुख', 'shahrukh', 'सलमान', 'salman',
                'आमिर', 'aamir', 'फिल्म', 'movie', 'अभिनेता', 'actor', 'सिनेमा', 'cinema',
                'गाना', 'song', 'म्यूजिक', 'music', 'अवार्ड', 'award'
            ],
            Domain.COMMUNITY: [
                'हिंदू', 'hindu', 'मुस्लिम', 'muslim', 'ईसाई', 'christian', 'सिख', 'sikh',
                'जाति', 'caste', 'धर्म', 'religion', 'समुदाय', 'community', 'संप्रदाय', 'sect',
                'आरक्षण', 'reservation', 'दलित', 'dalit', 'आदिवासी', 'tribal'
            ]
        }
        
        # Neural network layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, num_domains)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def keyword_based_classification(self, text: str) -> Dict[Domain, float]:
        """Rule-based classification using domain keywords"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by number of keywords
            domain_scores[domain] = score / len(keywords) if keywords else 0.0
        
        return domain_scores
    
    def neural_classification(self, embeddings: torch.Tensor) -> Tuple[Domain, float]:
        """Neural network based classification"""
        # Pool embeddings (mean pooling)
        pooled = torch.mean(embeddings, dim=1)  # [batch_size, embed_dim]
        
        # Get domain predictions
        domain_logits = self.classifier(pooled)
        domain_probs = torch.softmax(domain_logits, dim=-1)
        
        # Get confidence
        confidence = self.confidence_estimator(pooled)
        
        # Get predicted domain
        predicted_idx = torch.argmax(domain_probs, dim=-1)
        predicted_domain = self.domains[predicted_idx.item()]
        max_confidence = confidence.item()
        
        return predicted_domain, max_confidence
    
    def classify_query(self, text: str, embeddings: Optional[torch.Tensor] = None) -> Dict:
        """
        Classify query into appropriate domain
        
        Args:
            text: Input text query
            embeddings: Optional pre-computed embeddings from BharatVerify Transformer
        
        Returns:
            Classification results with domain, confidence, and reasoning
        """
        # Rule-based classification
        keyword_scores = self.keyword_based_classification(text)
        
        # Neural classification if embeddings provided
        neural_result = None
        if embeddings is not None:
            neural_domain, neural_confidence = self.neural_classification(embeddings)
            neural_result = {
                'domain': neural_domain,
                'confidence': neural_confidence
            }
        
        # Combine results (weighted approach)
        if neural_result and neural_result['confidence'] > 0.7:
            # High confidence neural prediction
            final_domain = neural_result['domain']
            final_confidence = neural_result['confidence']
            reasoning = "Neural network classification (high confidence)"
        elif keyword_scores:
            # Use rule-based if neural confidence is low
            final_domain = max(keyword_scores, key=keyword_scores.get)
            final_confidence = keyword_scores[final_domain]
            reasoning = "Keyword-based classification"
        else:
            # Default to general
            final_domain = Domain.GENERAL
            final_confidence = 0.5
            reasoning = "Default classification (no clear indicators)"
        
        return {
            'domain': final_domain,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'keyword_scores': keyword_scores,
            'neural_result': neural_result,
            'requires_specialized_engine': final_confidence > 0.6
        }

# Example usage and testing
def demo_domain_classification():
    """Demonstrate domain classification on sample Indian queries"""
    
    classifier = DomainClassifier()
    
    test_queries = [
        "प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है",  # Politics
        "क्या गाय का मूत्र कोविड का इलाज है?",  # Health  
        "भारत की जीडीपी 5 ट्रिलियन डॉलर हो गई है",  # Economics
        "विराट कोहली ने 100 शतक लगाए हैं",  # Sports
        "सभी मुसलमान आतंकवादी हैं",  # Community (sensitive)
        "शाहरुख खान की नई फिल्म हिट है",  # Entertainment
    ]
    
    print("🎯 Domain Classification Demo:")
    print("=" * 50)
    
    for query in test_queries:
        result = classifier.classify_query(query)
        
        print(f"\n📝 Query: {query}")
        print(f"🎯 Domain: {result['domain'].value}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"💡 Reasoning: {result['reasoning']}")
        print(f"⚙️ Needs Specialized Engine: {result['requires_specialized_engine']}")
        
        if result['keyword_scores']:
            top_domains = sorted(result['keyword_scores'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print(f"🔍 Top Keyword Matches:")
            for domain, score in top_domains:
                if score > 0:
                    print(f"   • {domain.value}: {score:.3f}")

if __name__ == "__main__":
    demo_domain_classification()
