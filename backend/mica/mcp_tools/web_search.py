"""
MCP Tool: Web Search
Description: Searches the web for information, with strong prioritization of federal government documents
Inputs: query (str), num_results (int), site_filter (list[str])
Outputs: List of search results with titles, URLs, and snippets, sorted by source authority

AGENT_INSTRUCTIONS:
You are a web search agent specialized in finding information from authoritative sources,
particularly federal government documents and official publications. Your task is to:

1. Formulate effective search queries based on the user's information needs
2. STRONGLY prioritize .gov domains and official sources in ALL searches
3. Search for technical reports, assessments, and official documents
4. Extract and summarize relevant information from search results
5. Identify the most credible and relevant sources

When searching for critical materials information, focus on:
- DOE reports and assessments (energy.gov)
- USGS mineral commodity summaries (usgs.gov)
- Trade data from Commerce/Census (commerce.gov, census.gov)
- EPA environmental assessments (epa.gov)
- Congressional Research Service reports
- IEA (International Energy Agency) reports
- World Bank data and reports

IMPORTANT: Always verify source credibility and prefer primary sources over secondary reporting.
News articles and market commentary should be de-prioritized in favor of official data.
Results are automatically scored and ranked by source authority.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus, urlparse

import httpx
import requests

from ..config import config
from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()


@register_tool
class WebSearchTool(MCPTool):
    """
    Web search tool for finding information from the web.

    Supports multiple search backends:
    - DuckDuckGo (default, no API key required)
    - Tavily (requires TAVILY_API_KEY)
    - SerpAPI (requires SERPAPI_API_KEY)

    Automatically scores and prioritizes results from official sources.
    """

    name = "web_search"
    description = "Search the web for information, with strong prioritization of federal government documents"
    version = "1.1.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    # Federal government domains to prioritize (highest authority)
    FEDERAL_DOMAINS = [
        "energy.gov",
        "doe.gov",
        "usgs.gov",
        "commerce.gov",
        "census.gov",
        "epa.gov",
        "congress.gov",
        "gao.gov",
        "whitehouse.gov",
        "state.gov",
        "treasury.gov",
        "trade.gov",
        "defense.gov",
        "nist.gov",
        "nsf.gov",
        "nasa.gov",
    ]

    # International authoritative sources (high authority)
    INTERNATIONAL_AUTHORITY_DOMAINS = [
        "iea.org",           # International Energy Agency
        "worldbank.org",     # World Bank
        "imf.org",           # IMF
        "oecd.org",          # OECD
        "un.org",            # United Nations
        "wto.org",           # World Trade Organization
    ]

    # Academic and research institutions (good authority)
    ACADEMIC_DOMAINS = [
        ".edu",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "wiley.com",
        "researchgate.net",
    ]

    # Industry/professional sources (moderate authority)
    INDUSTRY_DOMAINS = [
        "mining.com",
        "mining-technology.com",
        "spglobal.com",
        "woodmac.com",
        "mckinsey.com",
    ]

    # Low authority sources (news, blogs, stock sites)
    LOW_AUTHORITY_PATTERNS = [
        "msn.com",
        "yahoo.com",
        "reddit.com",
        "twitter.com",
        "facebook.com",
        "seekingalpha.com",
        "investopedia.com",
        "fool.com",
        "benzinga.com",
        "marketwatch.com",
        "zacks.com",
        "stocknews.com",
        "medium.com",
        "substack.com",
    ]

    # Source authority scores (higher = more authoritative)
    SOURCE_SCORES = {
        "federal": 100,
        "international": 85,
        "academic": 70,
        "industry": 50,
        "general": 30,
        "low_authority": 10,
    }

    def __init__(
        self,
        session_logger: Optional[SessionLogger] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the web search tool.

        Args:
            session_logger: Optional session logger
            provider: Search provider ('duckduckgo', 'tavily', 'serpapi')
        """
        super().__init__(session_logger)
        self.provider = provider or config.search.search_provider

    def _score_source(self, url: str) -> tuple[int, str]:
        """
        Score a source URL based on its authority level.

        Args:
            url: The URL to score

        Returns:
            Tuple of (score, category) where higher scores = more authoritative
        """
        url_lower = url.lower()

        # Check federal sources (highest priority)
        for domain in self.FEDERAL_DOMAINS:
            if domain in url_lower:
                return (self.SOURCE_SCORES["federal"], "federal")

        # Check international authority sources
        for domain in self.INTERNATIONAL_AUTHORITY_DOMAINS:
            if domain in url_lower:
                return (self.SOURCE_SCORES["international"], "international")

        # Check academic sources
        for domain in self.ACADEMIC_DOMAINS:
            if domain in url_lower:
                return (self.SOURCE_SCORES["academic"], "academic")

        # Check industry sources
        for domain in self.INDUSTRY_DOMAINS:
            if domain in url_lower:
                return (self.SOURCE_SCORES["industry"], "industry")

        # Check low authority sources
        for pattern in self.LOW_AUTHORITY_PATTERNS:
            if pattern in url_lower:
                return (self.SOURCE_SCORES["low_authority"], "low_authority")

        # Default score for general sources
        return (self.SOURCE_SCORES["general"], "general")

    def _rank_results(self, results: list[dict]) -> list[dict]:
        """
        Rank search results by source authority, placing official sources first.

        Args:
            results: List of search result dictionaries

        Returns:
            Sorted list with authoritative sources first
        """
        for result in results:
            url = result.get("url", "")
            score, category = self._score_source(url)
            result["authority_score"] = score
            result["source_category"] = category

        # Sort by authority score (descending)
        sorted_results = sorted(results, key=lambda x: x.get("authority_score", 0), reverse=True)

        logger.info(
            f"Ranked {len(sorted_results)} results: "
            f"{sum(1 for r in sorted_results if r.get('source_category') == 'federal')} federal, "
            f"{sum(1 for r in sorted_results if r.get('source_category') == 'international')} international, "
            f"{sum(1 for r in sorted_results if r.get('source_category') == 'academic')} academic"
        )

        return sorted_results

    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute a web search.

        Args:
            input_data: Dictionary with:
                - query (str): Search query
                - num_results (int, optional): Number of results (default 10)
                - site_filter (list[str], optional): Domains to restrict search to
                - federal_only (bool, optional): Only search federal domains

        Returns:
            ToolResult with list of search results
        """
        start_time = datetime.now()

        query = input_data.get("query")
        if not query:
            return ToolResult.error("Query is required")

        num_results = input_data.get("num_results", 10)
        site_filter = input_data.get("site_filter", [])
        federal_only = input_data.get("federal_only", False)

        # Apply federal domain filter if requested
        if federal_only:
            site_filter = self.FEDERAL_DOMAINS

        try:
            # Execute search based on provider
            if self.provider == "tavily":
                results = self._search_tavily(query, num_results, site_filter)
            elif self.provider == "serpapi":
                results = self._search_serpapi(query, num_results, site_filter)
            else:
                results = self._search_duckduckgo(query, num_results, site_filter)

            # Always rank results by source authority
            results = self._rank_results(results)

            # If we have very few official sources, do a supplementary search
            federal_count = sum(1 for r in results if r.get("source_category") == "federal")
            if federal_count < 2 and not federal_only and not site_filter:
                logger.info(f"Low federal source count ({federal_count}), performing supplementary .gov search")
                supplementary_results = self._search_federal_supplement(query, num_results=5)
                if supplementary_results:
                    # Merge and re-rank
                    existing_urls = {r.get("url") for r in results}
                    for supp in supplementary_results:
                        if supp.get("url") not in existing_urls:
                            results.append(supp)
                    results = self._rank_results(results)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Log search if session logger available
            if self.session_logger:
                search_id = hashlib.md5(query.encode()).hexdigest()[:8]
                self.session_logger.save_search_result(
                    search_id,
                    {
                        "query": query,
                        "provider": self.provider,
                        "num_results": len(results),
                        "results": results,
                        "federal_count": sum(1 for r in results if r.get("source_category") == "federal"),
                    },
                )

            return ToolResult(
                status=ToolResult.success(None).status,
                data=results,
                message=f"Found {len(results)} results ({sum(1 for r in results if r.get('source_category') in ['federal', 'international'])} from official sources)",
                execution_time=execution_time,
                metadata={
                    "query": query,
                    "provider": self.provider,
                    "num_results": len(results),
                    "federal_count": sum(1 for r in results if r.get("source_category") == "federal"),
                },
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult.error(str(e))

    def _enhance_query_for_authority(self, query: str) -> str:
        """
        Enhance a query to improve chances of finding authoritative sources.

        Adds terms that help surface government reports and official data.

        Args:
            query: The original search query

        Returns:
            Enhanced query string
        """
        query_lower = query.lower()

        # Detect if this is a critical materials query
        critical_materials_keywords = [
            "rare earth", "ree", "lithium", "cobalt", "nickel", "graphite",
            "critical mineral", "supply chain", "mining", "processing",
            "magnet", "battery", "semiconductor", "tungsten", "gallium",
            "germanium", "indium", "tantalum", "neodymium", "praseodymium",
        ]

        is_materials_query = any(kw in query_lower for kw in critical_materials_keywords)

        if is_materials_query:
            # Add terms that help surface official reports
            enhancement_terms = []

            # Add report type terms
            if "investment" in query_lower or "cost" in query_lower or "capital" in query_lower:
                enhancement_terms.append("(report OR assessment OR study)")
            elif "production" in query_lower or "capacity" in query_lower:
                enhancement_terms.append("(statistics OR data OR USGS)")
            elif "supply chain" in query_lower:
                enhancement_terms.append("(DOE OR report OR assessment)")
            else:
                enhancement_terms.append("(report OR data)")

            if enhancement_terms:
                return f"{query} {' '.join(enhancement_terms)}"

        return query

    def _search_federal_supplement(self, query: str, num_results: int = 5) -> list[dict]:
        """
        Perform a supplementary search specifically targeting federal .gov sources.

        This is called when the main search returns too few official sources.

        Args:
            query: The search query
            num_results: Number of results to fetch

        Returns:
            List of search results from federal sources
        """
        # Key federal domains for critical materials
        priority_domains = [
            "usgs.gov",
            "energy.gov",
            "doe.gov",
            "commerce.gov",
            "epa.gov",
            "gao.gov",
        ]

        all_results = []

        # Search with site restrictions for top federal sources
        site_query = " OR ".join([f"site:{domain}" for domain in priority_domains[:3]])
        federal_query = f"({query}) ({site_query})"

        try:
            from duckduckgo_search import DDGS
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            with DDGS() as ddgs:
                search_results = list(ddgs.text(federal_query, max_results=num_results))

                for item in search_results:
                    result = {
                        "title": item.get("title", ""),
                        "url": item.get("href", item.get("link", item.get("url", ""))),
                        "snippet": item.get("body", item.get("snippet", item.get("excerpt", ""))),
                        "source": urlparse(item.get("href", item.get("link", item.get("url", "")))).netloc,
                    }
                    # Score this result
                    score, category = self._score_source(result["url"])
                    result["authority_score"] = score
                    result["source_category"] = category
                    all_results.append(result)

            logger.info(f"Federal supplement search returned {len(all_results)} results")

        except Exception as e:
            logger.warning(f"Federal supplement search failed: {e}")

        return all_results

    def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        site_filter: list[str],
    ) -> list[dict]:
        """
        Search using DuckDuckGo via the duckduckgo-search package.

        Automatically enhances queries for critical materials topics to
        improve chances of finding authoritative sources.
        """
        # Build query with site restrictions
        if site_filter:
            site_query = " OR ".join([f"site:{domain}" for domain in site_filter])
            full_query = f"({query}) ({site_query})"
        else:
            # Enhance query for critical materials topics
            full_query = self._enhance_query_for_authority(query)

        try:
            # Try using the duckduckgo-search package (more reliable)
            from duckduckgo_search import DDGS
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            results = []
            with DDGS() as ddgs:
                # Try text search first
                search_results = list(ddgs.text(full_query, max_results=num_results))

                # If text search returns nothing, try news search as fallback
                if not search_results:
                    logger.info(f"Text search empty, trying news search for: {query[:50]}")
                    search_results = list(ddgs.news(full_query, max_results=num_results))

                for item in search_results:
                    result = {
                        "title": item.get("title", ""),
                        "url": item.get("href", item.get("link", item.get("url", ""))),
                        "snippet": item.get("body", item.get("snippet", item.get("excerpt", ""))),
                        "source": urlparse(item.get("href", item.get("link", item.get("url", "")))).netloc,
                    }
                    results.append(result)

            logger.info(f"DuckDuckGo search returned {len(results)} results for: {query[:50]}")
            return results

        except ImportError:
            logger.warning("duckduckgo-search package not installed, falling back to HTML scraping")
            return self._search_duckduckgo_html(full_query, num_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

    def _search_duckduckgo_html(self, query: str, num_results: int) -> list[dict]:
        """Fallback HTML scraping method for DuckDuckGo."""
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}

        try:
            response = requests.post(url, data=params, timeout=30)
            response.raise_for_status()
            return self._parse_ddg_html(response.text, num_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML fallback failed: {e}")
            return []

    def _parse_ddg_html(self, html: str, max_results: int) -> list[dict]:
        """Parse DuckDuckGo HTML results."""
        results = []
        import re

        link_pattern = r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</a>'

        links = re.findall(link_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(links[:max_results]):
            result = {
                "title": title.strip(),
                "url": url,
                "snippet": snippets[i] if i < len(snippets) else "",
                "source": urlparse(url).netloc,
            }
            results.append(result)

        return results

    def _search_tavily(
        self,
        query: str,
        num_results: int,
        site_filter: list[str],
    ) -> list[dict]:
        """Search using Tavily API."""
        if not config.tavily.is_configured:
            raise ValueError("Tavily API key not configured")

        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}

        payload = {
            "api_key": config.tavily.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": "advanced",
        }

        if site_filter:
            payload["include_domains"] = site_filter

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": urlparse(item.get("url", "")).netloc,
                "score": item.get("score"),
            })

        return results

    def _search_serpapi(
        self,
        query: str,
        num_results: int,
        site_filter: list[str],
    ) -> list[dict]:
        """Search using SerpAPI."""
        if not config.serpapi.is_configured:
            raise ValueError("SerpAPI key not configured")

        # Build query with site restrictions
        if site_filter:
            site_query = " OR ".join([f"site:{domain}" for domain in site_filter])
            full_query = f"({query}) ({site_query})"
        else:
            full_query = query

        url = "https://serpapi.com/search"
        params = {
            "api_key": config.serpapi.api_key,
            "q": full_query,
            "num": num_results,
            "engine": "google",
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": urlparse(item.get("link", "")).netloc,
                "position": item.get("position"),
            })

        return results


class FederalDocumentSearchTool(WebSearchTool):
    """
    Specialized search tool for federal government documents.

    Automatically restricts searches to federal government domains.
    """

    name = "federal_doc_search"
    description = "Search federal government documents and official publications"

    def execute(self, input_data: dict) -> ToolResult:
        """Execute a federal document search."""
        # Force federal domains
        input_data["federal_only"] = True
        return super().execute(input_data)
