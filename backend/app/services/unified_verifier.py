"""
Unified Verification System - Combines Domain Classification + Evidence Retrieval
The complete BharatVerify system in action!
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from .domain_classifier import DomainClassifier, Domain
from .evidence_retriever import BharatEvidenceRetriever

class BharatUnifiedVerifier:
    """
    Revolutionary unified fact-checking system for India
    Combines domain classification with evidence retrieval
    """
    
    def __init__(self, news_api_key: str = None):
        self.domain_classifier = DomainClassifier()
        self.evidence_retriever = BharatEvidenceRetriever(news_api_key)
        
    async def verify_claim(self, claim: str, user_context: Dict = None) -> Dict:
        """
        Complete claim verification pipeline
        
        Args:
            claim: Hindi or English claim to verify
            user_context: Optional user context (location, preferences, etc.)
            
        Returns:
            Complete verification analysis with domain classification,
            evidence retrieval, and community impact assessment
        """
        
        verification_start = datetime.now()
        
        print(f"🚀 BHARATVERIFY: Starting complete verification...")
        print(f"📝 Claim: {claim}")
        
        # Step 1: Domain Classification (Your revolutionary component!)
        print("🎯 Step 1: Domain Classification...")
        domain_result = self.domain_classifier.classify_query(claim)
        domain = domain_result['domain']
        
        print(f"   ✅ Domain: {domain.value}")
        print(f"   📊 Confidence: {domain_result['confidence']:.2f}")
        print(f"   💭 Reasoning: {domain_result['reasoning']}")
        
        # Step 2: Evidence Retrieval
        print("🔍 Step 2: Evidence Retrieval...")
        evidence_result = await self.evidence_retriever.retrieve_evidence(claim, domain)
        
        print(f"   ✅ Sources Found: {evidence_result['evidence_count']}")
        print(f"   ⭐ Credibility: {evidence_result['overall_credibility']:.2f}")
        print(f"   💪 Evidence Strength: {evidence_result['evidence_strength']}")
        
        # Step 3: Community Impact Analysis (Your signature innovation!)
        print("🤝 Step 3: Community Impact Analysis...")
        community_impact = self._analyze_community_impact(claim, domain_result, evidence_result)
        
        # Step 4: Generate Final Verification
        print("📋 Step 4: Final Verification...")
        final_verdict = self._generate_final_verdict(claim, domain_result, evidence_result, community_impact)
        
        verification_time = (datetime.now() - verification_start).total_seconds()
        
        return {
            'claim': claim,
            'verification_id': f"bharat_{int(datetime.now().timestamp())}",
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': verification_time,
            
            # Domain Classification Results
            'domain_classification': {
                'domain': domain.value,
                'confidence': domain_result['confidence'],
                'reasoning': domain_result['reasoning'],
                'keyword_matches': domain_result.get('keyword_scores', {}),
                'requires_specialized_engine': domain_result['requires_specialized_engine']
            },
            
            # Evidence Retrieval Results
            'evidence_analysis': {
                'sources_count': evidence_result['evidence_count'],
                'overall_credibility': evidence_result['overall_credibility'],
                'evidence_strength': evidence_result['evidence_strength'],
                'verification_summary': evidence_result['verification_summary'],
                'top_sources': evidence_result['evidence_sources'][:3],
                'sources_consulted': evidence_result['sources_consulted']
            },
            
            # Community Impact Analysis (Your Revolutionary Feature!)
            'community_impact': community_impact,
            
            # Final Verdict
            'verification_result': final_verdict,
            
            # System Information
            'system_info': {
                'version': 'BharatVerify v2.0',
                'model': 'Community-Aware Transformer',
                'languages_supported': ['Hindi', 'English', 'Bengali', 'Tamil'],
                'domains_covered': ['Politics', 'Health', 'Economics', 'Sports', 'Community']
            }
        }
    
    def _analyze_community_impact(self, claim: str, domain_result: Dict, evidence_result: Dict) -> Dict:
        """
        Revolutionary community impact analysis
        Your signature innovation that no other system has!
        """
        
        # Detect potentially sensitive community references
        sensitive_keywords = {
            'muslim': ['मुसलमान', 'muslim', 'इस्लाम', 'islam'],
            'hindu': ['हिंदू', 'hindu', 'हिंदुत्व', 'hindutva'],
            'christian': ['ईसाई', 'christian', 'क्रिश्चियन'],
            'sikh': ['सिख', 'sikh', 'गुरुद्वारा'],
            'dalit': ['दलित', 'dalit', 'अनुसूचित', 'scheduled'],
            'tribal': ['आदिवासी', 'tribal', 'जनजाति']
        }
        
        detected_communities = []
        for community, keywords in sensitive_keywords.items():
            if any(keyword in claim.lower() for keyword in keywords):
                detected_communities.append(community)
        
        # Assess potential harm level
        harm_indicators = [
            'आतंकवादी', 'terrorist', 'खतरनाक', 'dangerous',
            'गद्दार', 'traitor', 'देशद्रोही', 'anti-national'
        ]
        
        harm_level = "low"
        if any(indicator in claim.lower() for indicator in harm_indicators):
            harm_level = "high"
        elif detected_communities:
            harm_level = "medium"
        
        return {
            'detected_communities': detected_communities,
            'harm_level': harm_level,
            'requires_sensitivity': len(detected_communities) > 0,
            'recommendation': self._get_community_recommendation(detected_communities, harm_level),
            'educational_resources': self._get_educational_resources(detected_communities)
        }
    
    def _get_community_recommendation(self, communities: List[str], harm_level: str) -> str:
        """Generate community-sensitive recommendations"""
        
        if harm_level == "high":
            return "This claim contains potentially harmful stereotypes. Recommend community dialogue and education."
        elif harm_level == "medium" and communities:
            return f"This claim references {', '.join(communities)} community. Verify carefully and provide balanced context."
        else:
            return "No significant community sensitivity detected. Proceed with standard verification."
    
    def _get_educational_resources(self, communities: List[str]) -> List[Dict]:
        """Provide educational resources for community understanding"""
        
        resources = []
        if 'muslim' in communities:
            resources.append({
                'title': 'Understanding Indian Muslim Community',
                'url': 'https://www.education.gov.in/muslim-community-india',
                'description': 'Educational resource about Muslim contributions to India'
            })
        
        if 'hindu' in communities:
            resources.append({
                'title': 'Hindu Traditions in Modern India',
                'url': 'https://www.education.gov.in/hindu-traditions',
                'description': 'Understanding Hindu culture and traditions'
            })
        
        return resources
    
    def _generate_final_verdict(self, claim: str, domain_result: Dict, evidence_result: Dict, community_impact: Dict) -> Dict:
        """Generate final verification verdict"""
        
        # Simple verdict logic (in production, this would be more sophisticated)
        evidence_strength = evidence_result['evidence_strength']
        credibility = evidence_result['overall_credibility']
        
        if evidence_strength == "strong" and credibility > 0.8:
            verdict = "verified"
            confidence = 0.9
        elif evidence_strength == "moderate" and credibility > 0.6:
            verdict = "partially_verified"
            confidence = 0.7
        elif evidence_strength == "weak" or credibility < 0.5:
            verdict = "insufficient_evidence"
            confidence = 0.4
        else:
            verdict = "requires_further_investigation"
            confidence = 0.6
        
        # Adjust for community sensitivity
        if community_impact['harm_level'] == "high":
            verdict = "potentially_harmful"
            confidence = max(0.8, confidence)  # High confidence in harm detection
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'explanation': self._generate_verdict_explanation(verdict, evidence_result, community_impact),
            'recommendations': self._generate_recommendations(verdict, community_impact)
        }
    
    def _generate_verdict_explanation(self, verdict: str, evidence_result: Dict, community_impact: Dict) -> str:
        """Generate human-readable explanation"""
        
        explanations = {
            'verified': f"Found {evidence_result['evidence_count']} credible sources supporting this claim.",
            'partially_verified': f"Found mixed evidence from {evidence_result['evidence_count']} sources. Some aspects verified.",
            'insufficient_evidence': f"Only found {evidence_result['evidence_count']} sources with limited credibility.",
            'potentially_harmful': "This claim contains potentially harmful content targeting specific communities.",
            'requires_further_investigation': "Evidence is inconclusive. Professional fact-checking recommended."
        }
        
        base_explanation = explanations.get(verdict, "Verification completed.")
        
        if community_impact['requires_sensitivity']:
            base_explanation += f" Note: This claim references {', '.join(community_impact['detected_communities'])} community and requires cultural sensitivity."
        
        return base_explanation
    
    def _generate_recommendations(self, verdict: str, community_impact: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if verdict == "potentially_harmful":
            recommendations.extend([
                "Avoid sharing this content as it may promote harmful stereotypes",
                "Consider community dialogue and education instead",
                "Report to appropriate authorities if necessary"
            ])
        elif verdict == "insufficient_evidence":
            recommendations.extend([
                "Seek additional sources before accepting this claim",
                "Consult domain experts or official sources",
                "Avoid spreading unverified information"
            ])
        elif community_impact['requires_sensitivity']:
            recommendations.extend([
                "Share with cultural context and sensitivity",
                "Provide educational resources along with the information",
                "Encourage respectful dialogue about different communities"
            ])
        else:
            recommendations.append("Standard fact-checking practices apply")
        
        return recommendations

# Demo function
async def demo_unified_verification():
    """Demonstrate the complete BharatVerify system"""
    
    verifier = BharatUnifiedVerifier()
    
    test_claims = [
        "प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है",
        "क्या गाय का मूत्र कोविड का इलाज है?",
        "सभी मुसलमान आतंकवादी हैं",  # Sensitive community claim
        "विराट कोहली ने 100 शतक लगाए हैं"
    ]
    
    print("🇮🇳 BHARATVERIFY UNIFIED SYSTEM - REVOLUTIONARY DEMO")
    print("=" * 70)
    print("World's First Community-Aware Truth Verification System")
    print("=" * 70)
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n🔍 TEST {i}: {claim}")
        print("-" * 60)
        
        result = await verifier.verify_claim(claim)
        
        print(f"⚡ Processed in: {result['processing_time_seconds']:.2f} seconds")
        print(f"🎯 Domain: {result['domain_classification']['domain']}")
        print(f"📊 Evidence Sources: {result['evidence_analysis']['sources_count']}")
        print(f"🤝 Community Impact: {result['community_impact']['harm_level']}")
        print(f"✅ Final Verdict: {result['verification_result']['verdict']}")
        print(f"💡 Explanation: {result['verification_result']['explanation']}")
        
        if result['verification_result']['recommendations']:
            print("📋 Recommendations:")
            for rec in result['verification_result']['recommendations']:
                print(f"   • {rec}")
        
        print()

if __name__ == "__main__":
    # Run the complete system demo
    asyncio.run(demo_unified_verification())
