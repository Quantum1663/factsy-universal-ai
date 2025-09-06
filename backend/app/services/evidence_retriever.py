"""
Evidence Retrieval Engine for BharatVerify Transformer
Connects to live Indian data sources for real-time fact-checking
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from urllib.parse import quote_plus

from .domain_classifier import Domain

class IndianEvidenceSource:
    """Base class for Indian evidence sources"""
    
    def __init__(self, name: str, base_url: str, api_key: str = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.credibility_score = 0.8  # Default credibility
        
    async def search(self, query: str, domain: Domain) -> List[Dict]:
        """Search this source for evidence"""
        raise NotImplementedError

class PIBSource(IndianEvidenceSource):
    """Press Information Bureau - Official Government Source"""
    
    def __init__(self):
        super().__init__(
            name="PIB (Press Information Bureau)",
            base_url="https://pib.gov.in",
            api_key=None
        )
        self.credibility_score = 0.95  # Highest credibility for government source
        
    async def search(self, query: str, domain: Domain) -> List[Dict]:
        """Search PIB press releases and official statements"""
        try:
            # PIB doesn't have a public API, so we'll simulate with web scraping approach
            # In production, you'd implement proper web scraping or use their RSS feeds
            
            # For demo, we'll return structured government-style responses
            if domain == Domain.POLITICS:
                return await self._search_political_statements(query)
            elif domain == Domain.ECONOMICS:
                return await self._search_economic_data(query)
            else:
                return []
                
        except Exception as e:
            print(f"Error searching PIB: {e}")
            return []
    
    async def _search_political_statements(self, query: str) -> List[Dict]:
        """Search for political statements in PIB"""
        # Simulated PIB responses - in production, scrape actual PIB data
        mock_responses = [
            {
                "title": "PM's Statement on Economic Growth",
                "content": "Prime Minister announced comprehensive economic reforms",
                "date": "2025-09-01",
                "url": "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1962847",
                "source": "PIB",
                "credibility": 0.95,
                "relevance_score": 0.8
            }
        ]
        return mock_responses
    
    async def _search_economic_data(self, query: str) -> List[Dict]:
        """Search for economic data in government sources"""
        mock_responses = [
            {
                "title": "GDP Growth Rate - Official Statistics",
                "content": "India's GDP growth rate as per latest government data",
                "date": "2025-08-15",
                "url": "https://pib.gov.in/economic-data",
                "source": "PIB Economic Division",
                "credibility": 0.95,
                "relevance_score": 0.9
            }
        ]
        return mock_responses

class NewsAPISource(IndianEvidenceSource):
    """Indian News Sources via NewsAPI"""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="Indian News Media",
            base_url="https://newsapi.org/v2",
            api_key=api_key
        )
        self.credibility_score = 0.75
        self.indian_sources = [
            'the-times-of-india',
            'the-hindu', 
            'the-indian-express',
            'ndtv',
            'india-today'
        ]
    
    async def search(self, query: str, domain: Domain) -> List[Dict]:
        """Search Indian news sources for recent articles"""
        try:
            # Translate Hindi query to English for news search
            english_query = self._translate_hindi_keywords(query)
            
            # Add domain-specific keywords
            domain_keywords = self._get_domain_keywords(domain)
            full_query = f"{english_query} {domain_keywords}"
            
            # Search recent news (last 30 days)
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            params = {
                'q': full_query,
                'sources': ','.join(self.indian_sources),
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'language': 'en'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_news_results(data.get('articles', []))
                    else:
                        return []
                        
        except Exception as e:
            print(f"Error searching news: {e}")
            return []
    
    def _translate_hindi_keywords(self, query: str) -> str:
        """Simple Hindi to English keyword translation"""
        translations = {
            'मोदी': 'Modi',
            'प्रधानमंत्री': 'Prime Minister',
            'अर्थव्यवस्था': 'economy',
            'जीडीपी': 'GDP',
            'कोविड': 'COVID',
            'कोरोना': 'coronavirus',
            'विराट': 'Virat',
            'कोहली': 'Kohli',
            'क्रिकेट': 'cricket',
            'भारत': 'India',
            'सरकार': 'government'
        }
        
        english_query = query
        for hindi, english in translations.items():
            english_query = english_query.replace(hindi, english)
        
        return english_query
    
    def _get_domain_keywords(self, domain: Domain) -> str:
        """Get additional keywords based on domain"""
        domain_keywords = {
            Domain.POLITICS: "government policy election parliament",
            Domain.HEALTH: "medical healthcare doctor hospital treatment",
            Domain.ECONOMICS: "economy GDP inflation RBI finance",
            Domain.SPORTS: "cricket IPL sports match tournament",
            Domain.ENTERTAINMENT: "bollywood movie film actor actress"
        }
        return domain_keywords.get(domain, "")
    
    def _process_news_results(self, articles: List[Dict]) -> List[Dict]:
        """Process news API results into standard format"""
        processed = []
        for article in articles[:5]:  # Top 5 results
            processed.append({
                "title": article.get('title', ''),
                "content": article.get('description', ''),
                "date": article.get('publishedAt', ''),
                "url": article.get('url', ''),
                "source": article.get('source', {}).get('name', 'News Source'),
                "credibility": self.credibility_score,
                "relevance_score": 0.7  # Would be calculated based on content similarity
            })
        return processed

class FactCheckerSource(IndianEvidenceSource):
    """Indian Fact-Checking Organizations"""
    
    def __init__(self):
        super().__init__(
            name="Indian Fact Checkers",
            base_url="https://www.factchecker.in",
            api_key=None
        )
        self.credibility_score = 0.9
        self.fact_checkers = [
            {'name': 'Alt News', 'url': 'https://www.altnews.in'},
            {'name': 'Boom Live', 'url': 'https://www.boomlive.in'},
            {'name': 'The Quint WebQoof', 'url': 'https://www.thequint.com/news/webqoof'},
            {'name': 'India Today Fact Check', 'url': 'https://www.indiatoday.in/fact-check'}
        ]
    
    async def search(self, query: str, domain: Domain) -> List[Dict]:
        """Search Indian fact-checking websites"""
        # In production, implement web scraping of fact-checking sites
        # For now, return mock fact-check results
        
        if "मुसलमान" in query or "muslim" in query.lower():
            return [{
                "title": "Fact Check: Communal Claims Analysis",
                "content": "Professional fact-checking analysis of community-related claims",
                "date": "2025-09-01",
                "url": "https://www.altnews.in/fact-check-communal-claims",
                "source": "Alt News",
                "credibility": 0.9,
                "relevance_score": 0.95,
                "verdict": "Misleading",
                "community_impact": "High risk of communal tension"
            }]
        
        return []

class BharatEvidenceRetriever:
    """Main Evidence Retrieval Engine for Indian Fact-Checking"""
    
    def __init__(self, news_api_key: str = None):
        self.sources = [
            PIBSource(),
            NewsAPISource(news_api_key) if news_api_key else None,
            FactCheckerSource()
        ]
        # Remove None sources
        self.sources = [s for s in self.sources if s is not None]
        
    async def retrieve_evidence(self, claim: str, domain: Domain, max_sources: int = 10) -> Dict:
        """
        Retrieve evidence for a claim from multiple Indian sources
        
        Args:
            claim: The claim to fact-check (in Hindi or English)
            domain: Domain classification from your domain classifier
            max_sources: Maximum number of evidence sources to return
            
        Returns:
            Comprehensive evidence package with sources, credibility, and analysis
        """
        
        print(f"🔍 Retrieving evidence for: {claim[:50]}...")
        print(f"📊 Domain: {domain.value}")
        
        # Search all sources in parallel
        all_evidence = []
        tasks = []
        
        for source in self.sources:
            task = source.search(claim, domain)
            tasks.append(task)
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ Error in source {self.sources[i].name}: {result}")
                continue
            
            if result:
                # Add source metadata
                for evidence in result:
                    evidence['source_type'] = self.sources[i].name
                    evidence['retrieved_at'] = datetime.now().isoformat()
                
                all_evidence.extend(result)
        
        # Sort by relevance and credibility
        sorted_evidence = sorted(
            all_evidence,
            key=lambda x: (x.get('relevance_score', 0) + x.get('credibility', 0)) / 2,
            reverse=True
        )[:max_sources]
        
        # Calculate overall credibility
        if sorted_evidence:
            avg_credibility = sum(e.get('credibility', 0) for e in sorted_evidence) / len(sorted_evidence)
            evidence_strength = self._calculate_evidence_strength(sorted_evidence, domain)
        else:
            avg_credibility = 0.0
            evidence_strength = "insufficient"
        
        return {
            'claim': claim,
            'domain': domain.value,
            'evidence_count': len(sorted_evidence),
            'evidence_sources': sorted_evidence,
            'overall_credibility': avg_credibility,
            'evidence_strength': evidence_strength,
            'verification_summary': self._generate_verification_summary(claim, sorted_evidence, domain),
            'retrieved_at': datetime.now().isoformat(),
            'sources_consulted': [source.name for source in self.sources]
        }
    
    def _calculate_evidence_strength(self, evidence: List[Dict], domain: Domain) -> str:
        """Calculate overall strength of evidence"""
        
        if len(evidence) == 0:
            return "insufficient"
        elif len(evidence) >= 3 and any(e.get('credibility', 0) > 0.9 for e in evidence):
            return "strong"
        elif len(evidence) >= 2:
            return "moderate"
        else:
            return "weak"
    
    def _generate_verification_summary(self, claim: str, evidence: List[Dict], domain: Domain) -> str:
        """Generate human-readable verification summary"""
        
        if not evidence:
            return f"Insufficient evidence found to verify this {domain.value} claim. More research needed."
        
        high_credibility = [e for e in evidence if e.get('credibility', 0) > 0.85]
        
        if high_credibility:
            return f"Found {len(high_credibility)} high-credibility sources addressing this {domain.value} claim. Evidence suggests further verification needed."
        else:
            return f"Found {len(evidence)} sources related to this {domain.value} claim with mixed credibility. Cross-verification recommended."

# Demo function to test evidence retrieval
async def demo_evidence_retrieval():
    """Demonstrate evidence retrieval with Hindi queries"""
    
    retriever = BharatEvidenceRetriever()
    
    test_claims = [
        ("प्रधानमंत्री मोदी ने कहा कि भारत की अर्थव्यवस्था मजबूत है", Domain.POLITICS),
        ("क्या गाय का मूत्र कोविड का इलाज है?", Domain.HEALTH),
        ("विराट कोहली ने 100 शतक लगाए हैं", Domain.SPORTS),
    ]
    
    print("🔍 BHARAT EVIDENCE RETRIEVER - REVOLUTIONARY DEMO")
    print("=" * 60)
    
    for claim, domain in test_claims:
        print(f"\n📝 Claim: {claim}")
        print(f"🎯 Domain: {domain.value}")
        print("-" * 50)
        
        # Retrieve evidence
        evidence_result = await retriever.retrieve_evidence(claim, domain)
        
        print(f"📊 Evidence Sources Found: {evidence_result['evidence_count']}")
        print(f"⭐ Overall Credibility: {evidence_result['overall_credibility']:.2f}")
        print(f"💪 Evidence Strength: {evidence_result['evidence_strength']}")
        print(f"📋 Summary: {evidence_result['verification_summary']}")
        
        if evidence_result['evidence_sources']:
            print("🔗 Top Sources:")
            for i, source in enumerate(evidence_result['evidence_sources'][:3], 1):
                print(f"   {i}. {source['source_type']}: {source['title'][:60]}...")
                print(f"      Credibility: {source['credibility']:.2f} | Relevance: {source.get('relevance_score', 0):.2f}")
        
        print()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_evidence_retrieval())
