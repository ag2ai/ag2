#!/usr/bin/env python3
"""Newsroom Pipeline — multi-agent content creation through a Hub.

Category 3: Multiple Actors + Hub (Network of Agents)

Four specialized agents collaborate through a Hub to produce and publish
an article. The initiating agent discovers others dynamically and the
delegation chain flows: Researcher -> Writer -> Editor -> Publisher.

Usage:
    python playground/03_newsroom/main.py
    python playground/03_newsroom/main.py --scenario 2
    python playground/03_newsroom/main.py --model gemini-3-flash-preview
    python playground/03_newsroom/main.py "Write an article about Mars colonization"
"""

import asyncio
import sys
import time
from datetime import datetime

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolResultEvent
from autogen.beta.network import (
    Actor,
    DelegationRequest,
    DelegationResult,
    Hub,
    LoopDetector,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool

# ======================================================================
# ANSI formatting
# ======================================================================

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_WHITE = "\033[97m"

# ======================================================================
# Scenarios
# ======================================================================

SCENARIOS = {
    1: (
        "AI Agent Frameworks",
        "Write an article about the rise of AI agent frameworks and how "
        "they're changing software development",
    ),
    2: (
        "Crypto Mining Impact",
        "Write an investigative piece about the environmental impact of "
        "cryptocurrency mining in 2025",
    ),
    3: (
        "Quantum Computing",
        "Write a tech review of the latest advances in quantum computing "
        "and their practical applications",
    ),
}

# ======================================================================
# Tool definitions
# ======================================================================


def researcher_tools() -> list:
    @tool
    async def search_web(query: str) -> str:
        """Search the web for information on a topic.

        Args:
            query: Search query string.

        Returns:
            Search results with titles, snippets, and URLs.
        """
        results = {
            "ai agent": [
                {
                    "title": "The Rise of AI Agent Frameworks in 2025",
                    "snippet": "AI agent frameworks like AG2, LangGraph, and CrewAI are transforming how developers build autonomous systems. The market grew 340% in the past year.",
                    "url": "https://techreview.example.com/ai-agents-2025",
                },
                {
                    "title": "Enterprise Adoption of Multi-Agent Systems",
                    "snippet": "Fortune 500 companies report 67% improvement in workflow automation using multi-agent architectures. Key use cases: customer service, code generation, data analysis.",
                    "url": "https://enterprise-ai.example.com/multi-agent-adoption",
                },
                {
                    "title": "Agent Frameworks: Developer Survey 2025",
                    "snippet": "Survey of 12,000 developers shows 73% plan to integrate agent frameworks this year. Top concerns: reliability (42%), cost (31%), observability (27%).",
                    "url": "https://devsurvey.example.com/agents-2025",
                },
                {
                    "title": "From Chatbots to Autonomous Agents: The Paradigm Shift",
                    "snippet": "The transition from prompt-response chatbots to autonomous agents marks the biggest shift in AI application design since the transformer architecture.",
                    "url": "https://airesearch.example.com/paradigm-shift",
                },
            ],
            "cryptocurrency mining": [
                {
                    "title": "Crypto Mining's Carbon Footprint in 2025",
                    "snippet": "Despite the shift to proof-of-stake for some chains, Bitcoin mining still consumes 145 TWh annually. New study finds mining operations in Texas and Kazakhstan are the largest contributors.",
                    "url": "https://energywatch.example.com/crypto-carbon-2025",
                },
                {
                    "title": "Green Mining Initiatives Fall Short",
                    "snippet": "Industry pledges to reach 50% renewable energy by 2025 missed targets. Only 38% of mining operations use renewable sources, down from a claimed 58%.",
                    "url": "https://greentech.example.com/mining-renewables",
                },
                {
                    "title": "Water Consumption of Data Centers Surges",
                    "snippet": "Mining-heavy data centers in the US consumed 4.7 billion gallons of water in 2024 for cooling. Communities near mining operations report declining water tables.",
                    "url": "https://waterresources.example.com/datacenter-impact",
                },
            ],
            "quantum computing": [
                {
                    "title": "IBM and Google Achieve 1000+ Qubit Processors",
                    "snippet": "Both companies crossed the 1000-qubit milestone in 2025. IBM's Condor processor demonstrated quantum advantage for materials simulation.",
                    "url": "https://quantumweekly.example.com/1000-qubits",
                },
                {
                    "title": "Quantum Error Correction Breakthrough",
                    "snippet": "Microsoft Research achieves logical error rates below 0.1% using topological qubits. This threshold makes practical quantum computing feasible for the first time.",
                    "url": "https://msresearch.example.com/error-correction",
                },
                {
                    "title": "First Commercial Quantum Applications Emerge",
                    "snippet": "Pharmaceutical company Novartis used quantum simulation to identify 3 drug candidates in weeks instead of months. Financial firms report 10x speedup in portfolio optimization.",
                    "url": "https://quantumbiz.example.com/commercial-apps",
                },
                {
                    "title": "The Quantum Talent Gap",
                    "snippet": "Industry faces acute shortage of quantum engineers. Universities report 5x increase in quantum computing enrollment, but demand still outpaces supply.",
                    "url": "https://workforce.example.com/quantum-talent",
                },
            ],
        }
        # Find best matching results based on query keywords
        query_lower = query.lower()
        for key, entries in results.items():
            if key in query_lower:
                lines = []
                for i, r in enumerate(entries, 1):
                    lines.append(
                        f"[{i}] {r['title']}\n"
                        f"    {r['snippet']}\n"
                        f"    Source: {r['url']}"
                    )
                return "\n\n".join(lines)
        # Fallback: return the first set
        first_key = next(iter(results))
        entries = results[first_key]
        lines = []
        for i, r in enumerate(entries, 1):
            lines.append(
                f"[{i}] {r['title']}\n"
                f"    {r['snippet']}\n"
                f"    Source: {r['url']}"
            )
        return "\n\n".join(lines)

    @tool
    async def analyze_sources(topic: str, sources: str) -> str:
        """Analyze gathered sources and extract key findings.

        Args:
            topic: The topic being researched.
            sources: The raw source material to analyze.

        Returns:
            Structured analysis with key findings, quotes, and statistics.
        """
        return (
            f"Source Analysis for: {topic}\n"
            f"{'=' * 40}\n\n"
            f"Key Findings:\n"
            f"  1. The field is experiencing rapid growth with significant industry adoption\n"
            f"  2. Technical breakthroughs are enabling practical applications for the first time\n"
            f"  3. Challenges remain around scalability, cost, and workforce readiness\n\n"
            f"Notable Statistics:\n"
            f"  - Market growth exceeding 300% year-over-year\n"
            f"  - Enterprise adoption rate at 67% among Fortune 500\n"
            f"  - Developer interest at all-time high per survey data\n\n"
            f"Key Quotes:\n"
            f'  - "This marks the biggest paradigm shift since the transformer architecture"\n'
            f'  - "Practical applications are finally emerging from the lab"\n\n'
            f"Credibility: Sources include peer-reviewed research, industry surveys,\n"
            f"and verified reporting from established tech publications.\n\n"
            f"Recommendation: Strong enough basis for a feature article. Multiple\n"
            f"independent sources confirm the central thesis."
        )

    return [search_web, analyze_sources]


def writer_tools() -> list:
    @tool
    async def draft_article(
        headline: str, key_points: str, tone: str = "informative", word_count: int = 400
    ) -> str:
        """Draft an article based on research findings.

        Args:
            headline: The article headline.
            key_points: Key points and findings to include.
            tone: Writing tone (informative/investigative/conversational).
            word_count: Target word count for the article.

        Returns:
            A complete article draft.
        """
        return (
            f"# {headline}\n\n"
            f"*By Newsroom AI Staff | {datetime.now().strftime('%B %d, %Y')}*\n\n"
            f"The landscape is shifting beneath our feet. What was once the domain "
            f"of research labs and academic papers has burst into the mainstream, "
            f"reshaping industries and redefining what software can accomplish "
            f"on its own.\n\n"
            f"According to recent industry data, the field has experienced explosive "
            f"growth — over 300% year-over-year — driven by breakthroughs that are "
            f"making once-theoretical capabilities practical for the first time. "
            f"Enterprise adoption is accelerating, with 67% of Fortune 500 companies "
            f"now actively integrating these technologies into their workflows. "
            f"The results have been dramatic: workflow automation improvements, faster "
            f"time-to-market, and capabilities that simply were not possible "
            f"a year ago.\n\n"
            f"But the revolution is not without its growing pains. Developer surveys "
            f"reveal persistent concerns around reliability, cost management, and the "
            f"ability to observe and debug complex autonomous systems. As one researcher "
            f'noted, "This marks the biggest paradigm shift since the transformer '
            f'architecture" — and like that earlier shift, the full implications are '
            f"still unfolding.\n\n"
            f"What is clear is that the trajectory is set. With developer interest at "
            f"historic highs and practical applications emerging across sectors — from "
            f"healthcare to finance to software engineering itself — the question is "
            f"no longer whether this technology will transform the industry, but how "
            f"quickly organizations can adapt to the new reality.\n\n"
            f"---\n"
            f"*[Tone: {tone} | Target: {word_count} words]*"
        )

    @tool
    async def check_word_count(text: str) -> str:
        """Count the words in a text.

        Args:
            text: The text to count words in.

        Returns:
            Word count and brief assessment.
        """
        count = len(text.split())
        return f"Word count: {count} words. {'Good length.' if 200 <= count <= 600 else 'Consider adjusting length.'}"

    return [draft_article, check_word_count]


def editor_tools() -> list:
    @tool
    async def review_article(article_text: str) -> str:
        """Review an article and provide editorial feedback.

        Args:
            article_text: The full article text to review.

        Returns:
            Editorial feedback with suggestions and issues.
        """
        return (
            "Editorial Review\n"
            "================\n\n"
            "Overall Assessment: STRONG — publishable with minor revisions\n\n"
            "Strengths:\n"
            "  + Compelling opening that hooks the reader\n"
            "  + Good use of data points to support claims\n"
            "  + Balanced perspective — acknowledges challenges\n"
            "  + Clear narrative arc from past to present to future\n\n"
            "Suggestions:\n"
            "  - Consider adding a specific company case study for concreteness\n"
            "  - The third paragraph could use a stronger transition\n"
            "  - Final paragraph is strong but could benefit from a forward-looking quote\n\n"
            "Issues Found:\n"
            "  - Minor: one instance of passive voice that could be stronger\n"
            "  - Minor: 'simply were not possible' — consider rephrasing\n\n"
            "Verdict: Approve with minor edits. Ready for fact-check and publication."
        )

    @tool
    async def check_grammar(text: str) -> str:
        """Run a grammar check on the text.

        Args:
            text: Text to check for grammar issues.

        Returns:
            Grammar check results.
        """
        return (
            "Grammar Check Results\n"
            "---------------------\n"
            "Issues found: 2 (minor)\n\n"
            "1. Line 3: 'on its own' — consider 'autonomously' for more formal tone\n"
            "2. Line 8: comma splice detected — suggest splitting into two sentences\n\n"
            "Readability Score: 72/100 (Grade 10 level — appropriate for tech audience)\n"
            "Spelling: No errors found\n"
            "Tone consistency: PASS"
        )

    @tool
    async def verify_facts(claims: str) -> str:
        """Verify factual claims made in the article.

        Args:
            claims: Key claims to fact-check.

        Returns:
            Fact-check results for each claim.
        """
        return (
            "Fact Verification Report\n"
            "========================\n\n"
            "Claim: '300% year-over-year growth'\n"
            "  Status: VERIFIED — Multiple industry reports confirm 300-340% growth range\n\n"
            "Claim: '67% of Fortune 500 companies'\n"
            "  Status: VERIFIED — Enterprise AI adoption survey (Jan 2025) confirms 67%\n\n"
            "Claim: 'Developer interest at historic highs'\n"
            "  Status: VERIFIED — Stack Overflow and GitHub trend data support this\n\n"
            "Claim: 'biggest paradigm shift since the transformer'\n"
            "  Status: OPINION — attributed quote, no factual claim to verify\n\n"
            "Overall: All verifiable claims check out. Article is factually sound."
        )

    return [review_article, check_grammar, verify_facts]


def publisher_tools() -> list:
    @tool
    async def format_for_web(article_text: str, headline: str) -> str:
        """Format an article as a web-ready HTML snippet.

        Args:
            article_text: The finalized article text.
            headline: The article headline.

        Returns:
            HTML-formatted article snippet.
        """
        return (
            f'<article class="feature-story">\n'
            f"  <header>\n"
            f"    <h1>{headline}</h1>\n"
            f'    <p class="byline">By Newsroom AI Staff | '
            f'{datetime.now().strftime("%B %d, %Y")}</p>\n'
            f'    <p class="category">Technology</p>\n'
            f"  </header>\n"
            f'  <div class="article-body">\n'
            f"    {article_text[:200]}...\n"
            f"  </div>\n"
            f'  <footer class="article-meta">\n'
            f'    <span class="read-time">4 min read</span>\n'
            f"  </footer>\n"
            f"</article>\n\n"
            f"Formatted successfully. SEO tags, Open Graph metadata, and\n"
            f"responsive layout applied."
        )

    @tool
    async def generate_social_posts(headline: str, summary: str) -> str:
        """Generate social media posts for article promotion.

        Args:
            headline: The article headline.
            summary: Brief summary for social context.

        Returns:
            Social media posts for multiple platforms.
        """
        return (
            f"Social Media Posts Generated\n"
            f"============================\n\n"
            f"Twitter/X:\n"
            f'  "{headline} -- Our latest deep dive into the trends reshaping tech. '
            f'Read now: [link] #Tech #AI #Innovation"\n\n'
            f"LinkedIn:\n"
            f"  We just published a new feature: {headline}.\n"
            f"  {summary[:120]}...\n"
            f"  What do you think — is this the future? Read the full piece: [link]\n\n"
            f"Bluesky:\n"
            f'  New article: "{headline}"\n'
            f"  {summary[:100]}...\n"
            f"  [link]"
        )

    @tool
    async def schedule_publication(content: str, publish_time: str = "immediate") -> str:
        """Schedule an article for publication.

        Args:
            content: Reference to the formatted content.
            publish_time: When to publish (immediate / ISO datetime).

        Returns:
            Publication confirmation with details.
        """
        pub_id = f"PUB-{datetime.now().strftime('%Y%m%d')}-001"
        now = datetime.now().strftime("%H:%M:%S")
        return (
            f"Publication Scheduled\n"
            f"=====================\n"
            f"ID: {pub_id}\n"
            f"Time: {'NOW (' + now + ')' if publish_time == 'immediate' else publish_time}\n"
            f"Status: LIVE\n"
            f"Distribution: Website, RSS feed, newsletter queue\n"
            f"Social posts: Queued for auto-publish at T+5min, T+2hr, T+24hr\n"
            f"Analytics: Tracking pixel active, engagement dashboard live\n\n"
            f"Article is now published and live on the website."
        )

    return [format_for_web, generate_social_posts, schedule_publication]


# ======================================================================
# Agent creation
# ======================================================================


def create_researcher(model: str) -> Actor:
    return Actor(
        "researcher",
        prompt=(
            "You are a Senior Research Journalist.\n\n"
            "Your job is to research a topic thoroughly, then hand off your findings.\n"
            "Follow these steps:\n"
            "1. Use search_web to find relevant sources on the topic\n"
            "2. Use analyze_sources to extract key findings from what you found\n"
            "3. Use discover_agents to find an agent with writing capability\n"
            "4. Use delegate_to to send your research findings to the writer agent — "
            "include the topic, key findings, statistics, and source URLs so the "
            "writer has everything needed to draft the article\n"
            "5. Return the final result from the writer\n\n"
            "Be thorough in research but concise in your delegation message."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=researcher_tools(),
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )


def create_writer(model: str) -> Actor:
    return Actor(
        "writer",
        prompt=(
            "You are a Staff Writer at a tech publication.\n\n"
            "Your job is to take research findings and produce a polished article draft.\n"
            "Follow these steps:\n"
            "1. Use draft_article to write the article based on the research provided\n"
            "2. Use check_word_count to verify the draft is a good length\n"
            "3. Use discover_agents to find an agent with editing capability\n"
            "4. Use delegate_to to send your article draft to the editor — include "
            "the full article text and headline so they can review it\n"
            "5. Return the final result from the editor\n\n"
            "Write in a professional, engaging tone suitable for a tech audience."
        ),
        config=GeminiConfig(model=model, temperature=0.5),
        tools=writer_tools(),
        observers=[LoopDetector(repeat_threshold=3)],
    )


def create_editor(model: str) -> Actor:
    return Actor(
        "editor",
        prompt=(
            "You are a Senior Editor responsible for quality and accuracy.\n\n"
            "Your job is to review, fact-check, and approve articles for publication.\n"
            "Follow these steps:\n"
            "1. Use review_article to get editorial feedback on the draft\n"
            "2. Use check_grammar to verify writing quality\n"
            "3. Use verify_facts to check all factual claims\n"
            "4. Use discover_agents to find an agent with publishing capability\n"
            "5. Use delegate_to to send the approved article to the publisher — "
            "include the full article text, headline, and your editorial approval\n"
            "6. Return the final result from the publisher\n\n"
            "Be rigorous but constructive. Only block publication for serious issues."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=editor_tools(),
        observers=[LoopDetector(repeat_threshold=3)],
    )


def create_publisher(model: str) -> Actor:
    return Actor(
        "publisher",
        prompt=(
            "You are the Publishing Manager responsible for distribution.\n\n"
            "Your job is to prepare approved articles for publication and distribution.\n"
            "Follow these steps:\n"
            "1. Use format_for_web to create the web-ready version\n"
            "2. Use generate_social_posts to create social media promotion\n"
            "3. Use schedule_publication to publish the article\n"
            "4. Return a complete publication summary covering:\n"
            "   - Publication status and ID\n"
            "   - Social media posts created\n"
            "   - Distribution channels\n\n"
            "This is the final step in the pipeline — provide a thorough summary."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=publisher_tools(),
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )


# ======================================================================
# Logging helpers
# ======================================================================


def log(msg: str, *, style: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  {_DIM}{ts}{_RESET}  {style}{msg}{_RESET}")


def subscribe_agent_logging(stream: MemoryStream, agent_name: str) -> None:
    """Subscribe to an agent's stream for live tool/response logging."""

    async def _on_event(event: object) -> None:
        if isinstance(event, ToolCallEvent):
            if event.name == "delegate_to":
                args = event.serialized_arguments
                target = args.get("agent_name", "?")
                task_preview = args.get("task", "")[:100]
                log(
                    f"{_MAGENTA}{_BOLD}DELEGATE {agent_name} -> {target}{_RESET}",
                )
                log(f"  {_MAGENTA}{task_preview}{'...' if len(args.get('task', '')) > 100 else ''}{_RESET}")

            elif event.name == "discover_agents":
                args = event.serialized_arguments
                cap = args.get("capability", "")
                label = f'capability="{cap}"' if cap else "(all)"
                log(f"{_CYAN}{_BOLD}DISCOVER{_RESET} {_CYAN}{agent_name} queries agents {label}{_RESET}")

            else:
                try:
                    args = event.serialized_arguments
                    parts = [
                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                        for k, v in args.items()
                    ]
                    args_str = ", ".join(parts)
                    if len(args_str) > 120:
                        args_str = args_str[:120] + "..."
                except Exception:
                    args_str = event.arguments[:120]
                log(
                    f"{_YELLOW}{_BOLD}TOOL{_RESET} {_YELLOW}{agent_name}.{event.name}"
                    f"{_RESET}({_DIM}{args_str}{_RESET})"
                )

        elif isinstance(event, ToolResultEvent):
            preview = event.content[:140].replace("\n", " | ")
            if event.name == "delegate_to":
                log(
                    f"{_MAGENTA}{_BOLD}DELEGATE RESULT <-{_RESET} "
                    f"{_DIM}{preview}{'...' if len(event.content) > 140 else ''}{_RESET}"
                )
            elif event.name == "discover_agents":
                log(
                    f"{_CYAN}{_BOLD}DISCOVER RESULT:{_RESET} "
                    f"{_DIM}{preview}{_RESET}"
                )
            else:
                log(
                    f"  {_DIM}-> {preview}{'...' if len(event.content) > 140 else ''}{_RESET}"
                )

        elif isinstance(event, ModelResponse) and event.content:
            preview = event.content[:160].replace("\n", " | ")
            log(
                f"{_GREEN}{_BOLD}RESPONSE [{agent_name}]:{_RESET} "
                f"{_GREEN}{preview}{'...' if len(event.content) > 160 else ''}{_RESET}"
            )

    stream.subscribe(_on_event)


def subscribe_hub_logging(hub: Hub) -> None:
    """Subscribe to the Hub's stream for delegation events."""

    async def _on_hub_event(event: object) -> None:
        if isinstance(event, DelegationRequest):
            log(
                f"{_BLUE}{_BOLD}HUB DELEGATE{_RESET} "
                f"{_BLUE}{event.source} -> {event.target}: "
                f"{event.task[:80]}{'...' if len(event.task) > 80 else ''}{_RESET}"
            )
        elif isinstance(event, DelegationResult):
            preview = event.result[:100].replace("\n", " | ")
            log(
                f"{_BLUE}{_BOLD}HUB RESULT{_RESET} "
                f"{_BLUE}{event.target} -> {event.source}: "
                f"{preview}{'...' if len(event.result) > 100 else ''}{_RESET}"
            )

    hub.stream.subscribe(_on_hub_event)


# ======================================================================
# Main
# ======================================================================


def print_header(scenario_title: str, message: str, model: str) -> None:
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}NEWSROOM PIPELINE{_RESET}  {_DIM}Multi-Agent Content Creation{_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_BOLD}Scenario:{_RESET}  {scenario_title}")
    print(f"  {_BOLD}Model:{_RESET}     {model}")
    print()
    print(f"  {_BOLD}Pipeline:{_RESET}")
    print(f"    {_CYAN}Researcher{_RESET} -> {_YELLOW}Writer{_RESET} -> "
          f"{_RED}Editor{_RESET} -> {_GREEN}Publisher{_RESET}")
    print()
    print(f"  {_BOLD}Topic:{_RESET}")
    # Word-wrap
    words = message.split()
    line = "    "
    for w in words:
        if len(line) + len(w) + 1 > 72:
            print(line)
            line = "    " + w
        else:
            line += " " + w if len(line) > 4 else w
    if len(line) > 4:
        print(line)
    print()
    print(f"  {_BOLD}Log legend:{_RESET}")
    print(f"    {_YELLOW}TOOL{_RESET}              domain tool call")
    print(f"    {_CYAN}DISCOVER{_RESET}          agent discovery query")
    print(f"    {_MAGENTA}DELEGATE{_RESET}          inter-agent delegation")
    print(f"    {_BLUE}HUB DELEGATE{_RESET}      hub-level routing event")
    print(f"    {_GREEN}RESPONSE{_RESET}          agent final response")
    print()
    print(f"  {_DIM}{'- ' * 30}{_RESET}")
    print()


async def main() -> None:
    # Parse arguments
    model = "gemini-3.1-pro-preview"
    scenario_num = 1
    custom_topic = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--scenario" and i + 1 < len(args):
            scenario_num = int(args[i + 1])
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            custom_topic = args[i]
            i += 1
        else:
            i += 1

    if custom_topic:
        scenario_title = "Custom Topic"
        message = custom_topic
    else:
        scenario_title, message = SCENARIOS[scenario_num]

    print_header(scenario_title, message, model)

    # Create agents
    researcher = create_researcher(model)
    writer = create_writer(model)
    editor = create_editor(model)
    publisher = create_publisher(model)

    # Create Hub and register agents
    hub = Hub(max_delegation_depth=5)
    subscribe_hub_logging(hub)

    await hub.register(
        researcher,
        capabilities=["research", "fact-finding"],
        description="Researches topics and gathers source material",
    )
    await hub.register(
        writer,
        capabilities=["writing", "drafting"],
        description="Writes article drafts from research findings",
    )
    await hub.register(
        editor,
        capabilities=["editing", "review", "fact-checking"],
        description="Reviews, edits, and fact-checks articles",
    )
    await hub.register(
        publisher,
        capabilities=["publishing", "formatting"],
        description="Formats, promotes, and publishes articles",
    )

    log(f"{_BOLD}Hub ready. {len(hub.agents)} agents registered.{_RESET}")
    log(f"{_BOLD}Starting pipeline with researcher...{_RESET}")
    print()

    # Run the pipeline
    t0 = time.monotonic()
    reply = await hub.ask(researcher, message)
    elapsed = time.monotonic() - t0

    await hub.close()

    # Print final result
    print()
    print(f"  {_DIM}{'- ' * 30}{_RESET}")
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_GREEN}{_BOLD}PIPELINE COMPLETE{_RESET}  {_DIM}({elapsed:.1f}s){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()

    result = reply.body or "(no content)"
    for line in result.split("\n"):
        print(f"  {line}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
