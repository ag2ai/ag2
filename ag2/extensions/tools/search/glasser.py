# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Glasser paid-data tools for AG2, using one key and pay-per-call billing.

Maintainer: glasserai
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/glasser/
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

import anyio
import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool


def _pending(result: dict[str, Any]) -> bool:
    return result.get("status") in ("QUEUED", "RUNNING") or result.get("result_status") == "pending"


def _object(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Glasser returned a non-object JSON response")
    return result


@dataclass(frozen=True, slots=True)
class _GlasserClient:
    api_key: str = field(repr=False)
    base_url: str
    timeout: float
    wait_timeout: float
    poll_interval: float

    async def execute(self, capability: str, payload: dict[str, Any], ctx: Context) -> ToolResult:
        # Scope an LLM's call ID to its stream: different conversations can reuse call IDs.
        key = (
            str(uuid5(NAMESPACE_URL, f"ag2-glasser:{ctx.stream.id}:{ctx.tool_call_id}:{capability}"))
            if ctx.tool_call_id
            else str(uuid4())
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "ag2-glasser-extension"},
            timeout=self.timeout,
            follow_redirects=False,
        ) as client:
            path = f"v1/solutions/gtm/{capability}"
            response = await client.post(
                path,
                json={key: value for key, value in payload.items() if value is not None},
                headers={"Idempotency-Key": key},
            )
            result = _object(response)
            if not _pending(result) or self.wait_timeout == 0:
                return ToolResult(result)

            if capability == "people_search":
                # People search returns a native Run; the other capabilities have a result URL.
                run_id = result.get("id")
                if not isinstance(run_id, str) or not run_id or any(c not in "0123456789abcdefABCDEF-" for c in run_id):
                    raise ValueError("Glasser returned an invalid pending Run id")
                poll_url = client.base_url.join(f"v1/runs/{run_id}")
            else:
                result_url = result.get("result_url")
                if not isinstance(result_url, str):
                    raise ValueError("Glasser returned no result_url for a pending result")
                poll_url = client.base_url.join(result_url)
                expected = client.base_url.join(f"{path}/result")
                if (
                    (poll_url.scheme, poll_url.host, poll_url.port, poll_url.path)
                    != (expected.scheme, expected.host, expected.port, expected.path)
                    or poll_url.userinfo
                    or poll_url.fragment
                ):
                    raise ValueError("Glasser result_url must point to this capability on the configured API origin")

            # Only reads are repeated. On expiry retain the pending result and its retrieval information.
            with anyio.move_on_after(self.wait_timeout):
                while _pending(result):
                    await anyio.sleep(self.poll_interval)
                    result = _object(await client.get(poll_url))
            return ToolResult(result)


class GlasserToolkit(Toolkit):
    """Six paid-data tools backed by Glasser's GTM Solution API.

    Register the toolkit for all six tools, or use its factory methods to select
    a subset. Factory defaults can be fixed values or AG2 ``Variable`` markers.
    Results preserve the API's status, output, errors, charges and Run references.
    """

    __slots__ = ("_client", "_provider", "_limit", "_country", "_task_id", "_metadata")

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.glasser.ai",
        timeout: float = 60.0,
        wait_timeout: float = 120.0,
        poll_interval: float = 1.0,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        country: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit without making an HTTP request.

        Args:
            api_key: Glasser API key, sent as a Bearer credential.
            base_url: API root, or a compatible gateway URL.
            timeout: Positive HTTP timeout in seconds, per request.
            wait_timeout: Maximum seconds to poll an admitted execution; zero disables polling.
            poll_interval: Positive delay in seconds between result reads.
            provider: Default Provider; omitted means server-side automatic selection.
            limit: Default result count, where supported. Can affect the charge.
            country: Default country for web, company and SEO research.
            task_id: Default Task for grouping related Runs; not sent for social research.
            metadata: Tracking fields passed to Glasser, not to the data Provider.
            middleware: Middleware applied to all default tools.

        Raises:
            ValueError: If credentials, the API URL or timing settings are invalid.
        """
        if not api_key.strip():
            raise ValueError("api_key is required")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) API root without credentials, query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError("wait_timeout must be finite and non-negative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        self._client = _GlasserClient(api_key, base_url.rstrip("/") + "/", timeout, wait_timeout, poll_interval)
        self._provider = provider
        self._limit = limit
        self._country = country
        self._task_id = task_id
        self._metadata = metadata
        super().__init__(
            self.web_research(),
            self.company_intelligence(),
            self.people_search(),
            self.seo_research(),
            self.social_research(),
            self.market_data(),
            name="glasser_toolkit",
            middleware=middleware,
        )

    def web_research(
        self,
        *,
        provider: str | Variable | None = None,
        country: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_web_research",
        description: str = "Search the web, news, places, papers, shopping, images or videos; answer questions, fetch a URL or find similar pages. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the web research tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            country: Country; omitted inherits the toolkit default.
            limit: Result count; omitted inherits the toolkit default.
            task_id: Task for related Runs; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        country = self._country if country is None else country
        limit = self._limit if limit is None else limit
        task_id = self._task_id if task_id is None else task_id
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_web_research(
            ctx: Context,
            action: Annotated[
                Literal[
                    "search", "news", "places", "scholar", "shopping", "images", "videos", "answer", "scrape", "similar"
                ]
                | None,
                Field(
                    description="Default: search.\n\n- Supply query for search, news, places, scholar, shopping, images, videos, or answer.\n- Supply url for scrape or similar."
                ),
            ] = None,
            query: Annotated[str | None, Field(description="Search query.")] = None,
            url: Annotated[str | None, Field(description="URL to fetch or inspect.")] = None,
            formats: Annotated[
                list[Literal["text", "markdown", "html", "raw_html"]] | None,
                Field(
                    description="For scrape only. When omitted, use the selected Provider's native default: text for Serper/Exa, markdown for Firecrawl/DataForSEO. Requested formats guide routing and fallback; output exposes fixed text/markdown/html/raw_html fields. html means cleaned HTML; raw_html means original HTML. Auto uses Serper then Firecrawl. HTML requests go directly to Firecrawl. Read result_status and missing_formats for delivery; an unrequested null field is not a partial result. Firecrawl does not supply text; Glasser does not convert HTML to text. X/Twitter and enrichment routes are excluded from Firecrawl."
                ),
            ] = None,
            timeout_ms: Annotated[
                int | None,
                Field(
                    description="For scrape only. Default 30000. Execution budget for this API request, including serial fallback. A retry has a new request window and reuses existing Runs through Run idempotency. Firecrawl and Exa use the actual remaining time, reserving settlement and response time; less than 1 second of call time is refused."
                ),
            ] = None,
            max_charge_usd: Annotated[
                str | None,
                Field(
                    description="Exact USD decimal string (min 2, max 6 decimals, one canonical spelling). Sum with decimal arithmetic or scale by 10^6 — never floats."
                ),
            ] = None,
            cache_max_age_seconds: Annotated[
                int | None,
                Field(
                    description="For scrape only. Maximum Provider cache age; 0 requires a live fetch. Routes without this control are excluded. Auto selects Firecrawl (milliseconds); explicit Exa receives whole hours rounded down. Unknown cache/fetch metadata remains null."
                ),
            ] = None,
            time_range: Annotated[
                Literal["day", "week", "month", "year"] | None,
                Field(
                    description="For search only: restrict results to the past day, week, month, or year using Google's time filter. Omit for any time. Auto uses only Providers that support this filter (Serper and SerpApi). Exa is refused when this field is supplied. This is a search-engine filter, not a verified publication date."
                ),
            ] = None,
            language: Annotated[
                str | None,
                Field(
                    description="Language for search-type actions, two letters such as en, de. Not for scrape or similar."
                ),
            ] = None,
        ) -> ToolResult:
            """Search the web, news, places, papers, shopping, images or videos; answer questions, fetch a URL or find similar pages."""
            return await self._client.execute(
                "web_research",
                {
                    "action": action,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "query": query,
                    "url": url,
                    "formats": formats,
                    "timeout_ms": timeout_ms,
                    "max_charge_usd": max_charge_usd,
                    "cache_max_age_seconds": cache_max_age_seconds,
                    "time_range": time_range,
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "language": language,
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_web_research

    def company_intelligence(
        self,
        *,
        provider: str | Variable | None = None,
        country: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_company_intelligence",
        description: str = "Look up a company profile, technologies, traffic, competitors, funding or news by domain. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the company intelligence tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            country: Country; omitted inherits the toolkit default.
            limit: Result count; omitted inherits the toolkit default.
            task_id: Task for related Runs; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        country = self._country if country is None else country
        limit = self._limit if limit is None else limit
        task_id = self._task_id if task_id is None else task_id
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_company_intelligence(
            ctx: Context,
            domain: Annotated[
                str,
                Field(
                    description="A website domain such as stripe.com. A full URL is accepted and reduced to its host."
                ),
            ],
            action: Annotated[
                Literal["enrich", "tech_stack", "traffic", "competitors", "funding", "news"] | None,
                Field(
                    description="Default: enrich. All actions require domain.\n\n- enrich: company profile and firmographics.\n- tech_stack, traffic, funding, news: the corresponding company data.\n- competitors: similar businesses. For competing domains in Google results, use SEO research with organic_competitors."
                ),
            ] = None,
            query: Annotated[
                str | None,
                Field(
                    description="For news via Serper only: the company name to search Google News by, instead of the domain. Other actions and providers ignore it."
                ),
            ] = None,
        ) -> ToolResult:
            """Look up a company profile, technologies, traffic, competitors, funding or news by domain."""
            return await self._client.execute(
                "company_intelligence",
                {
                    "action": action,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "domain": domain,
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "query": query,
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_company_intelligence

    def people_search(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_people_search",
        description: str = "Search people by job or company, enrich a person, or find a work email. Returns a Provider-native Run; inspect provider_response. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the people search tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            limit: Result count; omitted inherits the toolkit default.
            task_id: Task for related Runs; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        limit = self._limit if limit is None else limit
        task_id = self._task_id if task_id is None else task_id
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_people_search(
            ctx: Context,
            action: Annotated[
                Literal["search", "enrich", "find_email"] | None,
                Field(
                    description="Default: search.\n\n- search: supply at least one of job_titles, seniorities, locations, company_domain, or keywords. Apollo and ZoomInfo rows carry no email or phone; PDL rows are full records; Hunter lists the personal addresses at company_domain.\n- enrich: supply linkedin_url, email, or full_name plus company_domain. Prospeo returns only people it holds a verified work email for.\n- find_email: supply full_name and company_domain. Apollo also accepts linkedin_url alone; an Apollo match without a real email is a miss and is not charged.\n\nFilters are never silently dropped. With auto, Providers that cannot apply them are skipped; a named Provider that cannot apply them returns 400.\n\nThe output is the answering Provider's native body, so its shape differs per Provider: read run.provider before parsing it."
                ),
            ] = None,
            job_titles: Annotated[
                list[str] | None,
                Field(
                    description="Job titles to match. PDL matches each title exactly as given, so 'CTO' does not match 'Co-founder & CTO'."
                ),
            ] = None,
            seniorities: Annotated[
                list[
                    Literal[
                        "owner",
                        "founder",
                        "c_suite",
                        "partner",
                        "vp",
                        "head",
                        "director",
                        "manager",
                        "senior",
                        "entry",
                        "intern",
                    ]
                ]
                | None,
                Field(
                    description="Seniority levels, one or more of owner, founder, c_suite, partner, vp, head, director, manager, senior, entry, intern. A Provider without a matching level returns 400 when named and is skipped under auto. ZoomInfo has no founder, owner, partner, head, entry or intern level; its senior is the whole Non Manager tier, which also includes junior individual contributors."
                ),
            ] = None,
            locations: Annotated[
                list[str] | None,
                Field(
                    description="Locations as free text. PDL matches each as a substring of the person's location name, so 'london' also matches 'new london, connecticut'."
                ),
            ] = None,
            company_domain: Annotated[
                str | None,
                Field(
                    description="A website domain such as stripe.com. A full URL is accepted and reduced to its host."
                ),
            ] = None,
            keywords: Annotated[str | None, Field(description="Keywords to research.")] = None,
            full_name: Annotated[str | None, Field(description="Person name.")] = None,
            email: Annotated[str | None, Field(description="An email address.")] = None,
            linkedin_url: Annotated[
                str | None,
                Field(description="The person's LinkedIn profile URL, linkedin.com/in/<slug>. Not a company page."),
            ] = None,
        ) -> ToolResult:
            """Search people by job or company, enrich a person, or find a work email. Returns a Provider-native Run; inspect provider_response."""
            return await self._client.execute(
                "people_search",
                {
                    "action": action,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "job_titles": job_titles,
                    "seniorities": seniorities,
                    "locations": locations,
                    "company_domain": company_domain,
                    "keywords": keywords,
                    "full_name": full_name,
                    "email": email,
                    "linkedin_url": linkedin_url,
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_people_search

    def seo_research(
        self,
        *,
        provider: str | Variable | None = None,
        country: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_seo_research",
        description: str = "Research keywords, rankings, competitors, backlinks, site audits and AI visibility. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the seo research tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            country: Country; omitted inherits the toolkit default.
            limit: Result count; omitted inherits the toolkit default.
            task_id: Task for related Runs; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        country = self._country if country is None else country
        limit = self._limit if limit is None else limit
        task_id = self._task_id if task_id is None else task_id
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_seo_research(
            ctx: Context,
            action: Annotated[
                Literal[
                    "keyword_overview",
                    "keyword_ideas",
                    "serp",
                    "domain_overview",
                    "ranked_keywords",
                    "organic_competitors",
                    "backlinks_overview",
                    "backlinks",
                    "referring_domains",
                    "domain_rating",
                    "site_audit",
                    "ai_visibility",
                    "keyword_gap",
                    "backlink_gap",
                    "page_audit",
                    "visibility_history",
                    "ai_citations",
                ]
                | None,
                Field(
                    description="Default: keyword_overview.\n\n- keyword_overview: supply one or more keywords.\n- keyword_ideas or serp: supply exactly one keyword in keywords.\n- site_audit: supply domain; limit is the number of pages to crawl (default 20, at most 100). The crawl runs asynchronously.\n- keyword_gap and backlink_gap: supply domain and one to four competitors.\n- page_audit: supply url (one page).\n- visibility_history: supply domain; limit is the number of months (default 12, at most 24).\n- All other actions: supply domain. organic_competitors finds domains ranking for the same keywords; ai_visibility: AI-answer visibility in the answering Provider's own metric (Ahrefs: citations of the domain's pages; DataForSEO: LLM mentions of the domain); ai_citations: pages most cited in AI answers that mention the domain, from any site.\n- keyword_gap rows differ by Provider: Semrush returns keywords any of the domains ranks for (union); DataForSEO returns keywords the competitor ranks for and you do not."
                ),
            ] = None,
            keywords: Annotated[list[str] | None, Field(description="Keywords to research.")] = None,
            domain: Annotated[
                str | None,
                Field(
                    description="A website domain such as stripe.com. A full URL is accepted and reduced to its host."
                ),
            ] = None,
            competitors: Annotated[
                list[str] | None,
                Field(
                    description="Competitor domains for keyword_gap and backlink_gap, one to four. Full URLs are reduced to their host."
                ),
            ] = None,
            url: Annotated[
                str | None, Field(description="One page URL for page_audit, such as https://stripe.com/pricing.")
            ] = None,
        ) -> ToolResult:
            """Research keywords, rankings, competitors, backlinks, site audits and AI visibility."""
            return await self._client.execute(
                "seo_research",
                {
                    "action": action,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "keywords": keywords,
                    "domain": domain,
                    "competitors": competitors,
                    "url": url,
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_seo_research

    def social_research(
        self,
        *,
        provider: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_social_research",
        description: str = "Search public posts, read profiles or feeds, retrieve a post, or find accounts on six social platforms. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the social research tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_social_research(
            ctx: Context,
            platform: Annotated[
                Literal["reddit", "x", "youtube", "tiktok", "instagram", "linkedin"],
                Field(description="Social platform."),
            ],
            mode: Annotated[
                Literal["search", "profile", "feed", "post", "find"] | None,
                Field(
                    description="Default: search. Every platform supports each mode.\n\n- search: posts matching query.\n- profile: account details from handle (Reddit: subreddit). For LinkedIn, supply a person or company url.\n- feed: recent posts or videos from handle. For LinkedIn, supply a company url.\n- post: one post or video from url.\n- find: other social profiles from handle. For LinkedIn, supply a person's profile url; for Reddit, supply query to find subreddits."
                ),
            ] = None,
            query: Annotated[str | None, Field(description="Search query.")] = None,
            handle: Annotated[
                str | None, Field(description="A username without @, or a subreddit name without r/.")
            ] = None,
            url: Annotated[str | None, Field(description="URL to fetch or inspect.")] = None,
        ) -> ToolResult:
            """Search public posts, read profiles or feeds, retrieve a post, or find accounts on six social platforms."""
            return await self._client.execute(
                "social_research",
                {
                    "platform": platform,
                    "mode": mode,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "query": query,
                    "handle": handle,
                    "url": url,
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_social_research

    def market_data(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        name: str = "glasser_market_data",
        description: str = "Read US property values, rents, property records, listings, ZIP market statistics or stock quotes. Paid per call. Inspect result_status (when present), status and charge_usd separately. A pending result is an existing execution; do not submit another paid request to poll it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the market data tool.

        Args:
            provider: Provider to select; omitted inherits the toolkit default.
            limit: Result count; omitted inherits the toolkit default.
            task_id: Task for related Runs; omitted inherits the toolkit default.
            metadata: Tracking fields; omitted inherits the toolkit default.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A tool that returns the full Glasser API result.
        """
        provider = self._provider if provider is None else provider
        limit = self._limit if limit is None else limit
        task_id = self._task_id if task_id is None else task_id
        metadata = self._metadata if metadata is None else metadata

        @tool(name=name, description=description, middleware=middleware)
        async def glasser_market_data(
            ctx: Context,
            action: Annotated[
                Literal[
                    "property_value",
                    "property_rent",
                    "property_search",
                    "listings_sale",
                    "listings_rental",
                    "market_stats",
                    "stock_quote",
                ],
                Field(
                    description="Choose the data to retrieve:\n\n- property_value or property_rent: supply address.\n- property_search, listings_sale, or listings_rental: supply address, city plus state, or zip.\n- market_stats: supply zip.\n- stock_quote: supply symbol."
                ),
            ],
            address: Annotated[str | None, Field(description="US street address.")] = None,
            city: Annotated[str | None, Field(description="US city.")] = None,
            state: Annotated[str | None, Field(description="Two-letter US state code.")] = None,
            zip: Annotated[str | None, Field(description="Five-digit US ZIP code.")] = None,
            symbol: Annotated[str | None, Field(description="Stock ticker symbol.")] = None,
        ) -> ToolResult:
            """Read US property values, rents, property records, listings, ZIP market statistics or stock quotes."""
            return await self._client.execute(
                "market_data",
                {
                    "action": action,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "address": address,
                    "city": city,
                    "state": state,
                    "zip": zip,
                    "symbol": symbol,
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
            )

        return glasser_market_data
