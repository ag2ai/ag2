"""Voice of Customer Intelligence — always-on social listening with AI-powered routing.

Category 11: Voice of Customer (VoC) — autonomous social listening network that
continuously monitors customer feedback across channels, classifies mentions using
AI, and routes actionable insights to specialist agents.

Six agents coordinate a real-time VoC pipeline: Collector scans social channels,
Analyst classifies and routes, and four specialist agents handle defects, competitive
intelligence, PR crises, and critical escalations. Custom observers track mention
volume spikes and sentiment drift in real-time.

Usage:
    python playground/11_voice_of_customer/main.py                    # product launch
    python playground/11_voice_of_customer/main.py --scenario 2       # crisis detection
    python playground/11_voice_of_customer/main.py --scenario 3       # autonomous monitoring
    python playground/11_voice_of_customer/main.py --scenario 4       # competitive intel
    python playground/11_voice_of_customer/main.py --model gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

from autogen.beta import (
    Actor,
    BaseObserver,
    BatchWatch,
    EventWatch,
    IntervalWatch,
    LoopDetector,
    ObserverAlert,
    Severity,
    TokenMonitor,
    tool,
)
from autogen.beta.annotations import Context
from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolResultEvent
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network import (
    BasePlugin,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    Envelope,
    HubContext,
    Network,
    Pipeline,
    SchedulerTriggerFired,
    TelemetryPlugin,
)

if TYPE_CHECKING:
    from autogen.beta.network.hub import Hub

# =====================================================================
# ANSI helpers
# =====================================================================

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
_BG_RED = "\033[41m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _severity_color(sev: str) -> str:
    sev_upper = sev.upper() if isinstance(sev, str) else str(sev).upper()
    if "CRITICAL" in sev_upper:
        return _RED
    if "WARNING" in sev_upper:
        return _YELLOW
    return _CYAN


# =====================================================================
# Simulated Social Media Data — NovaBand X1 smartwatch by NovaTech
# =====================================================================

_X_MENTIONS = [
    {
        "id": "x-1001",
        "author": "@techreviewer_mike",
        "followers": 12_400,
        "text": (
            "Day 3 with the NovaBand X1 and battery is already dying after 6 hours. "
            "Wasn't this supposed to last 3 days? Charging every night defeats the "
            "purpose of a fitness tracker. #NovaBandX1 #disappointed"
        ),
        "likes": 342,
        "retweets": 89,
        "timestamp": "2026-03-22T14:32:00Z",
    },
    {
        "id": "x-1002",
        "author": "@fitness_jenna",
        "followers": 3_200,
        "text": (
            "Loving my new NovaBand X1! The workout tracking is incredibly accurate "
            "and the GPS is spot-on for my morning runs. Best fitness watch I've owned. "
            "#NovaBandX1 #fitness"
        ),
        "likes": 87,
        "retweets": 12,
        "timestamp": "2026-03-22T09:15:00Z",
    },
    {
        "id": "x-1003",
        "author": "@gadget_guru_sam",
        "followers": 45_000,
        "text": (
            "Compared the NovaBand X1 side-by-side with Apple Watch Ultra 3. GPS is "
            "comparable but the X1 heart rate sensor lags 2-3 seconds behind. At half "
            "the price though, hard to complain. Full review dropping Friday. #smartwatch"
        ),
        "likes": 1_203,
        "retweets": 445,
        "timestamp": "2026-03-21T18:45:00Z",
    },
    {
        "id": "x-1004",
        "author": "@sarahk_running",
        "followers": 156_000,
        "text": (
            "SAFETY WARNING: My NovaBand X1 got extremely hot during charging last "
            "night — too hot to touch. Left a mark on my nightstand. @NovaTech this "
            "is a serious safety issue. Anyone else experiencing this? #NovaBandX1"
        ),
        "likes": 4_521,
        "retweets": 2_187,
        "timestamp": "2026-03-23T07:20:00Z",
    },
    {
        "id": "x-1005",
        "author": "@dave_techie",
        "followers": 890,
        "text": (
            "Wish the NovaBand X1 had a proper sleep tracking mode with REM detection. "
            "The current sleep data is too basic compared to what Fitbit offers. "
            "Hope NovaTech adds this in a firmware update. #NovaBandX1"
        ),
        "likes": 45,
        "retweets": 8,
        "timestamp": "2026-03-22T22:10:00Z",
    },
    {
        "id": "x-1006",
        "author": "@budget_buyer_77",
        "followers": 1_500,
        "text": (
            "$349 for the NovaBand X1 is insane when Samsung Galaxy Watch 7 does "
            "everything it does for $279. NovaTech needs a reality check on pricing. "
            "#overpriced #NovaBandX1"
        ),
        "likes": 678,
        "retweets": 234,
        "timestamp": "2026-03-23T11:00:00Z",
    },
    {
        "id": "x-1007",
        "author": "@wellness_maria",
        "followers": 8_400,
        "text": (
            "Getting a red, itchy rash on my wrist after wearing the NovaBand X1 for "
            "just 2 days. Had to stop wearing it. Is this a nickel allergy issue? "
            "Several others reporting the same. @NovaTech please respond. #NovaBandX1"
        ),
        "likes": 892,
        "retweets": 367,
        "timestamp": "2026-03-23T13:45:00Z",
    },
    {
        "id": "x-1008",
        "author": "@viral_vinny",
        "followers": 520_000,
        "text": (
            "Thread: I bought FIVE NovaBand X1 watches for my team. THREE have screen "
            "defects out of the box. One won't even turn on. @NovaTech this is the "
            "worst product launch I've seen in years. Full rant with photos below. "
            "1/12 🧵 #NovaBandX1 #NovaTechFail"
        ),
        "likes": 12_340,
        "retweets": 8_923,
        "timestamp": "2026-03-23T16:30:00Z",
    },
]

_REDDIT_POSTS = [
    {
        "id": "reddit-2001",
        "subreddit": "r/smartwatches",
        "author": "u/battery_drain_victim",
        "title": "NovaBand X1 battery drain — anyone else seeing 15% per hour?",
        "text": (
            "Got my X1 two days ago. Battery drains at roughly 15% per hour with "
            "always-on display. GPS tracking kills it in under 3 hours. Factory reset "
            "didn't help. NovaTech support said 'wait for firmware update' but this "
            "is unacceptable for a $349 device. Multiple others in the NovaTech forums "
            "reporting the same issue."
        ),
        "upvotes": 847,
        "comments": 234,
        "timestamp": "2026-03-22T20:15:00Z",
    },
    {
        "id": "reddit-2002",
        "subreddit": "r/gadgets",
        "author": "u/comparison_king",
        "title": "Honest comparison: NovaBand X1 vs Fitbit Sense 4 vs Galaxy Watch 7",
        "text": (
            "Bought all three to compare. NovaBand X1 wins on build quality and GPS "
            "accuracy. Fitbit Sense 4 has the best health ecosystem. Galaxy Watch 7 "
            "best for Android integration. X1 battery is worst of the three by far. "
            "If NovaTech fixes the battery, it's the best value. Until then, Galaxy "
            "Watch 7 is my pick for most people."
        ),
        "upvotes": 1_243,
        "comments": 567,
        "timestamp": "2026-03-21T15:30:00Z",
    },
    {
        "id": "reddit-2003",
        "subreddit": "r/wearables",
        "author": "u/happy_runner_42",
        "title": "NovaBand X1 is genuinely great for running",
        "text": (
            "3 weeks in and the X1 has become my go-to running watch. GPS locks in "
            "under 5 seconds, heart rate is accurate during intervals, and the VO2 "
            "max estimate is within 2% of my lab test. The running dynamics (cadence, "
            "ground contact time) are surprisingly good for the price point."
        ),
        "upvotes": 432,
        "comments": 89,
        "timestamp": "2026-03-20T10:00:00Z",
    },
    {
        "id": "reddit-2004",
        "subreddit": "r/smartwatches",
        "author": "u/charger_melted",
        "title": "WARNING: NovaBand X1 charging cradle melted overnight",
        "text": (
            "Woke up to a burning smell. The NovaBand X1 magnetic charger partially "
            "melted on my wooden nightstand. The watch itself has a scorch mark on "
            "the back. This is a genuine fire hazard. I've filed a complaint with "
            "the CPSC and contacted NovaTech. Photos in comments. DO NOT charge this "
            "overnight until NovaTech addresses this."
        ),
        "upvotes": 3_456,
        "comments": 1_289,
        "timestamp": "2026-03-23T08:45:00Z",
    },
    {
        "id": "reddit-2005",
        "subreddit": "r/legaladvice",
        "author": "u/burned_consumer",
        "title": "NovaBand X1 caused minor burn on my wrist — grounds for class action?",
        "text": (
            "The NovaBand X1 overheated during a workout and left a first-degree burn "
            "on my wrist. I have medical documentation and photos. I know at least 30 "
            "other people reporting skin burns or device overheating. Is there grounds "
            "for a class action? I'm in California."
        ),
        "upvotes": 2_109,
        "comments": 876,
        "timestamp": "2026-03-23T14:20:00Z",
    },
    {
        "id": "reddit-2006",
        "subreddit": "r/smartwatches",
        "author": "u/feature_wisher",
        "title": "NovaBand X1 feature wishlist — what NovaTech should add",
        "text": (
            "Love the hardware but the software needs work. My wishlist: 1) REM sleep "
            "tracking, 2) Custom watch faces marketplace, 3) Offline Spotify, 4) Blood "
            "pressure estimation, 5) Better notification management. The hardware can "
            "support all of this — NovaTech just needs to invest in the software."
        ),
        "upvotes": 567,
        "comments": 234,
        "timestamp": "2026-03-22T12:00:00Z",
    },
]

_REVIEWS = [
    {
        "id": "review-3001",
        "platform": "google_reviews",
        "author": "Michael T.",
        "rating": 1,
        "title": "Screen cracked after 2 weeks",
        "text": (
            "The screen developed a hairline crack after just 2 weeks of normal use. "
            "No drops, no impacts. Contacted support and they said it's not covered "
            "under warranty because they consider it 'physical damage'. Avoid."
        ),
        "timestamp": "2026-03-21T09:00:00Z",
    },
    {
        "id": "review-3002",
        "platform": "google_reviews",
        "author": "Jennifer L.",
        "rating": 5,
        "title": "Best fitness tracker I've owned",
        "text": (
            "Switched from Garmin to NovaBand X1 and couldn't be happier. The workout "
            "tracking is phenomenal, the display is gorgeous, and the health metrics "
            "are comprehensive. Battery could be better but I charge it during my "
            "morning shower. Not a dealbreaker."
        ),
        "timestamp": "2026-03-20T14:30:00Z",
    },
    {
        "id": "review-3003",
        "platform": "google_reviews",
        "author": "Robert K.",
        "rating": 2,
        "title": "Heart rate sensor is wildly inaccurate",
        "text": (
            "Compared the X1 heart rate readings with my chest strap during workouts. "
            "The X1 was off by 15-20 BPM during high-intensity intervals. For a device "
            "marketed for fitness, this is unacceptable. Resting HR is fine but "
            "exercise readings are useless."
        ),
        "timestamp": "2026-03-22T11:00:00Z",
    },
    {
        "id": "review-3004",
        "platform": "google_reviews",
        "author": "Amanda S.",
        "rating": 4,
        "title": "Good but overpriced compared to Samsung",
        "text": (
            "The NovaBand X1 is a solid smartwatch with great build quality. But at "
            "$349, it's hard to justify over the Samsung Galaxy Watch 7 at $279 which "
            "has better battery life and similar features. NovaTech needs to drop the "
            "price by $50 or add more value to stay competitive."
        ),
        "timestamp": "2026-03-21T16:00:00Z",
    },
    {
        "id": "review-3005",
        "platform": "app_store",
        "author": "FitnessFan2026",
        "rating": 1,
        "title": "Skin rash from the band",
        "text": (
            "Developed a painful, red, blistering rash on my wrist after 3 days of "
            "wearing the NovaBand X1. Dermatologist confirmed it's a contact dermatitis "
            "reaction, likely from nickel in the sensor housing. This should be a recall."
        ),
        "timestamp": "2026-03-23T10:30:00Z",
    },
    {
        "id": "review-3006",
        "platform": "app_store",
        "author": "CasualUser99",
        "rating": 3,
        "title": "Decent but needs more features",
        "text": (
            "It's a fine smartwatch for the basics — notifications, step counting, "
            "and heart rate. But compared to Apple Watch or even Fitbit, the smart "
            "features are lacking. No offline music, no voice assistant, limited "
            "third-party apps. Good hardware, mediocre software."
        ),
        "timestamp": "2026-03-22T08:00:00Z",
    },
]

_NEWS_ARTICLES = [
    {
        "id": "news-4001",
        "source": "TechCrunch",
        "headline": "NovaBand X1 Review: Ambitious hardware held back by software",
        "snippet": (
            "NovaTech's first smartwatch shows impressive hardware engineering — the "
            "GPS accuracy rivals Garmin, the AMOLED display is stunning, and the build "
            "quality exceeds its $349 price point. But battery life averaging 18 hours "
            "and sparse app ecosystem leave it trailing Apple and Samsung. Score: 7/10."
        ),
        "url": "techcrunch.com/2026/03/20/novaband-x1-review",
        "timestamp": "2026-03-20T12:00:00Z",
    },
    {
        "id": "news-4002",
        "source": "The Verge",
        "headline": "NovaBand X1 battery complaints are piling up fast",
        "snippet": (
            "Just days after launch, the NovaBand X1 is facing a wave of battery "
            "complaints. Users report 15-20% per hour drain with basic use, far below "
            "NovaTech's claimed 72-hour battery life. The company says a firmware fix "
            "is 'in testing' but hasn't provided a timeline. Some retailers are already "
            "seeing higher-than-normal return rates."
        ),
        "url": "theverge.com/2026/03/22/novaband-x1-battery-issues",
        "timestamp": "2026-03-22T16:00:00Z",
    },
    {
        "id": "news-4003",
        "source": "Consumer Reports",
        "headline": "NovaBand X1 overheating reports prompt safety investigation",
        "snippet": (
            "Consumer Reports has launched an investigation into the NovaBand X1 "
            "following reports of overheating during charging and skin burns during "
            "exercise. At least 47 consumers have filed complaints with the CPSC. "
            "NovaTech has not yet issued a formal response to safety concerns."
        ),
        "url": "consumerreports.org/2026/03/23/novaband-x1-safety",
        "timestamp": "2026-03-23T09:00:00Z",
    },
    {
        "id": "news-4004",
        "source": "Bloomberg",
        "headline": "NovaTech shares drop 8% as NovaBand X1 complaints go viral",
        "snippet": (
            "NovaTech Inc. (NVTK) shares fell 8.2% in early trading as social media "
            "complaints about its flagship NovaBand X1 smartwatch gained traction. A "
            "viral Twitter thread documenting quality defects has accumulated over 12K "
            "retweets. Analysts at Morgan Stanley downgraded the stock to 'underweight' "
            "citing potential recall costs."
        ),
        "url": "bloomberg.com/2026/03/23/novatech-stock-drop",
        "timestamp": "2026-03-23T15:00:00Z",
    },
]

_KNOWN_DEFECTS = {
    "DEF-001": {
        "component": "battery",
        "title": "Excessive battery drain under normal use",
        "severity": "high",
        "status": "investigating",
        "reported_count": 1_247,
        "firmware_affected": "v1.0.0 — v1.0.2",
    },
    "DEF-002": {
        "component": "display",
        "title": "Spontaneous screen cracking without impact",
        "severity": "medium",
        "status": "confirmed",
        "reported_count": 89,
        "firmware_affected": "all",
    },
    "DEF-003": {
        "component": "charging_cradle",
        "title": "Charging cradle overheating and melting",
        "severity": "critical",
        "status": "active_investigation",
        "reported_count": 47,
        "firmware_affected": "all (hardware issue)",
    },
    "DEF-004": {
        "component": "sensor_housing",
        "title": "Contact dermatitis from nickel in sensor housing",
        "severity": "high",
        "status": "confirmed",
        "reported_count": 312,
        "firmware_affected": "N/A (hardware/materials)",
    },
    "DEF-005": {
        "component": "heart_rate_sensor",
        "title": "Inaccurate heart rate during high-intensity exercise",
        "severity": "medium",
        "status": "investigating",
        "reported_count": 523,
        "firmware_affected": "v1.0.0 — v1.0.2",
    },
}

_COMPETITOR_DATA = {
    "Apple Watch Ultra 3": {
        "price": 799,
        "battery_life": "48 hours",
        "strengths": ["ecosystem integration", "health features", "app store", "build quality"],
        "weaknesses": ["price", "iOS only", "size/weight"],
        "market_share": "34%",
        "sentiment": "positive",
    },
    "Samsung Galaxy Watch 7": {
        "price": 279,
        "battery_life": "40 hours",
        "strengths": ["Android integration", "price/value", "battery life", "rotating bezel"],
        "weaknesses": ["health accuracy", "third-party app quality", "Wear OS bugs"],
        "market_share": "18%",
        "sentiment": "mixed-positive",
    },
    "Fitbit Sense 4": {
        "price": 299,
        "battery_life": "6 days",
        "strengths": ["battery life", "health ecosystem", "sleep tracking", "stress management"],
        "weaknesses": ["limited smart features", "aging design", "slow updates"],
        "market_share": "12%",
        "sentiment": "stable-positive",
    },
    "Garmin Venu 4": {
        "price": 449,
        "battery_life": "11 days",
        "strengths": ["battery life", "fitness accuracy", "durability", "offline maps"],
        "weaknesses": ["smart features", "display brightness", "price for features"],
        "market_share": "8%",
        "sentiment": "niche-positive",
    },
}


# =====================================================================
# Custom Observers
# =====================================================================


class VolumeTracker(BaseObserver):
    """Tracks mention volume by category and alerts on spikes.

    Fires every 3 tool calls on the analyst. Parses tool results
    to track how many mentions fall into each category. Warns if any
    single category (especially defects or safety) dominates the batch,
    suggesting the analyst should prioritize escalation.
    """

    SPIKE_THRESHOLD = 3  # warn if any category exceeds this in one batch
    SAFETY_THRESHOLD = 2  # lower threshold for safety-related categories

    def __init__(self) -> None:
        super().__init__(
            name="volume-tracker",
            watch=BatchWatch(n=3, condition=ToolCallEvent),
        )
        self._category_counts: dict[str, int] = {}
        self._total_analyzed: int = 0
        self._spikes_detected: int = 0
        self._warnings: int = 0

    @property
    def stats(self) -> dict:
        return {
            "total_analyzed": self._total_analyzed,
            "category_counts": dict(self._category_counts),
            "spikes_detected": self._spikes_detected,
            "warnings": self._warnings,
        }

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        batch_categories: dict[str, int] = {}

        for event in events:
            if not isinstance(event, ToolCallEvent):
                continue
            if event.name != "analyze_mention":
                continue
            self._total_analyzed += 1
            # Track based on tool call arguments — the analyst is classifying
            args_str = str(event.args) if hasattr(event, "args") else ""
            for kw, cat in [
                ("safety", "safety"),
                ("overheat", "safety"),
                ("burn", "safety"),
                ("rash", "safety"),
                ("defect", "defect"),
                ("battery", "defect"),
                ("crack", "defect"),
                ("broken", "defect"),
                ("competitor", "competitor"),
                ("pricing", "competitor"),
                ("viral", "pr_risk"),
                ("lawsuit", "legal"),
                ("class action", "legal"),
            ]:
                if kw in args_str.lower():
                    batch_categories[cat] = batch_categories.get(cat, 0) + 1
                    self._category_counts[cat] = self._category_counts.get(cat, 0) + 1
                    break

        # Check for safety spikes (lower threshold)
        safety_count = batch_categories.get("safety", 0) + batch_categories.get("legal", 0)
        if safety_count >= self.SAFETY_THRESHOLD:
            self._spikes_detected += 1
            self._warnings += 1
            return ObserverAlert(
                source=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"SAFETY SPIKE: {safety_count} safety/legal mentions in this batch. "
                    f"Total safety mentions: {self._category_counts.get('safety', 0)}. "
                    f"Prioritize routing these to the escalation team immediately. "
                    f"Consider bundling safety mentions for urgent ticket creation."
                ),
                data={
                    "batch_safety_count": safety_count,
                    "total_categories": dict(self._category_counts),
                },
            )

        # Check for general volume spikes
        for cat, count in batch_categories.items():
            if count >= self.SPIKE_THRESHOLD:
                self._spikes_detected += 1
                self._warnings += 1
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Volume spike: {count} '{cat}' mentions in this batch "
                        f"(threshold: {self.SPIKE_THRESHOLD}). Total {cat}: "
                        f"{self._category_counts.get(cat, 0)}. Consider routing a "
                        f"consolidated brief to the appropriate specialist."
                    ),
                    data={"category": cat, "batch_count": count},
                )

        return None


class SentimentMonitor(BaseObserver):
    """Tracks overall sentiment from analyst tool results.

    Fires on every ToolResultEvent. Parses sentiment scores from
    analyze_mention results and tracks a running average. Warns if
    sentiment trends strongly negative, signaling a potential crisis.
    """

    WARNING_THRESHOLD = -0.3
    CRITICAL_THRESHOLD = -0.5

    def __init__(self) -> None:
        super().__init__(
            name="sentiment-monitor",
            watch=EventWatch(ToolResultEvent),
        )
        self._scores: list[float] = []
        self._warned_moderate = False
        self._warned_severe = False
        self._warnings: int = 0

    @property
    def stats(self) -> dict:
        avg = sum(self._scores) / len(self._scores) if self._scores else 0.0
        return {
            "total_scored": len(self._scores),
            "average_sentiment": round(avg, 3),
            "warnings": self._warnings,
        }

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if not isinstance(event, ToolResultEvent):
                continue
            content = event.content or ""
            # Parse sentiment score from analyze_mention results
            if "sentiment_score" in content:
                try:
                    # Look for patterns like sentiment_score: -0.7
                    for part in content.split("\n"):
                        if "sentiment_score" in part.lower():
                            for token in part.split():
                                try:
                                    score = float(token.strip(",:()"))
                                    if -1.0 <= score <= 1.0:
                                        self._scores.append(score)
                                except ValueError:
                                    continue
                except Exception:
                    pass

        if len(self._scores) < 3:
            return None

        avg = sum(self._scores) / len(self._scores)

        # CRITICAL at -0.5
        if avg <= self.CRITICAL_THRESHOLD and not self._warned_severe:
            self._warned_severe = True
            self._warnings += 1
            return ObserverAlert(
                source=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"Severe negative sentiment: average {avg:.2f} across "
                    f"{len(self._scores)} mentions. Customer perception is strongly "
                    f"negative. Ensure PR response team is engaged and escalation "
                    f"handles any safety-critical items."
                ),
                data={"average_sentiment": round(avg, 3), "sample_size": len(self._scores)},
            )

        # WARNING at -0.3
        if avg <= self.WARNING_THRESHOLD and not self._warned_moderate:
            self._warned_moderate = True
            self._warnings += 1
            return ObserverAlert(
                source=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Negative sentiment trend: average {avg:.2f} across "
                    f"{len(self._scores)} mentions. More negative than positive "
                    f"customer feedback detected. Monitor closely and prioritize "
                    f"routing negative mentions for resolution."
                ),
                data={"average_sentiment": round(avg, 3), "sample_size": len(self._scores)},
            )

        return None


class SourceHealthCheck(BaseObserver):
    """Monitors data source reliability from collector tool results.

    Fires on every ToolResultEvent from collector search tools.
    Warns if any source returns zero results or error indicators,
    suggesting the source may be down or rate-limited.
    """

    def __init__(self) -> None:
        super().__init__(
            name="source-health-check",
            watch=EventWatch(ToolResultEvent),
        )
        self._source_status: dict[str, str] = {}
        self._empty_sources: int = 0
        self._warnings: int = 0

    @property
    def stats(self) -> dict:
        return {
            "source_status": dict(self._source_status),
            "empty_sources": self._empty_sources,
            "warnings": self._warnings,
        }

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if not isinstance(event, ToolResultEvent):
                continue
            content = event.content or ""
            source_name = event.name or ""

            # Only monitor search tools
            if not source_name.startswith("search_"):
                continue

            if "Found 0" in content or "No results" in content or "error" in content.lower():
                self._source_status[source_name] = "empty/error"
                self._empty_sources += 1
                self._warnings += 1
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Source '{source_name}' returned empty or errored results. "
                        f"This channel may be temporarily unavailable or rate-limited. "
                        f"Consider retrying later or relying on other sources for now."
                    ),
                    data={"tool": source_name, "status": "empty/error"},
                )

            self._source_status[source_name] = "healthy"

        return None


# =====================================================================
# ContentRouter — routing plugin (soft enforcement)
# =====================================================================


class ContentRouter(BasePlugin):
    """Validates analyst-to-specialist routing based on content category.

    Soft enforcement: logs warnings for unexpected routes but allows them.
    Ensures the analyst routes defects to product-inspector, competitor
    mentions to market-intel, PR risks to pr-responder, and safety/legal
    issues to escalation.

    Expected routes:
        collector       -> analyst
        analyst         -> product-inspector, market-intel, pr-responder, escalation
        product-inspector -> analyst (report back)
        market-intel      -> analyst (report back)
        pr-responder      -> analyst (report back)
        escalation        -> analyst (report back)
    """

    EXPECTED_ROUTES: dict[str, list[str]] = {
        "collector": ["analyst"],
        "analyst": ["product-inspector", "market-intel", "pr-responder", "escalation"],
        "product-inspector": ["analyst"],
        "market-intel": ["analyst"],
        "pr-responder": ["analyst"],
        "escalation": ["analyst"],
    }

    # Keyword-to-specialist mapping for deeper validation
    CATEGORY_ROUTING: dict[str, str] = {
        "defect": "product-inspector",
        "battery": "product-inspector",
        "quality": "product-inspector",
        "sensor": "product-inspector",
        "screen": "product-inspector",
        "competitor": "market-intel",
        "pricing": "market-intel",
        "market": "market-intel",
        "comparison": "market-intel",
        "viral": "pr-responder",
        "reputation": "pr-responder",
        "influencer": "pr-responder",
        "stock": "pr-responder",
        "safety": "escalation",
        "legal": "escalation",
        "burn": "escalation",
        "recall": "escalation",
        "cpsc": "escalation",
        "overheat": "escalation",
    }

    def __init__(self) -> None:
        super().__init__()
        self.total_routed = 0
        self.total_warnings = 0
        self.correct_routes = 0
        self.route_log: list[tuple[str, str, bool]] = []

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        source = envelope.sender or ""
        target = envelope.recipient or ""
        allowed_targets = self.EXPECTED_ROUTES.get(source, [])
        is_allowed = target in allowed_targets or source == "scheduler"

        self.total_routed += 1
        self.route_log.append((source, target, is_allowed))

        if is_allowed:
            self.correct_routes += 1

        if not is_allowed and source:
            self.total_warnings += 1
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_BLUE}{_BOLD}CONTENT ROUTER{_RESET} "
                f"{_YELLOW}Unexpected route: {source} -> {target} "
                f"(expected: {', '.join(allowed_targets) or 'none'}){_RESET}"
            )

        return envelope  # Soft enforcement — always allow


# =====================================================================
# InsightTracker — system plugin (observes hub stream)
# =====================================================================


class InsightTracker(BasePlugin):
    """Tracks VoC insights flowing through the network for reporting.

    System plugin: observes hub.stream without affecting routing.
    Builds a real-time dashboard of what categories are being processed,
    which specialists are active, and the overall throughput.
    """

    def __init__(self) -> None:
        super().__init__()
        self._hub: Hub | None = None
        self._sub_ids: list = []
        self.delegation_log: list[dict] = []
        self.specialist_activity: dict[str, int] = {}
        self.total_insights: int = 0

    def install(self, hub: Hub) -> None:  # type: ignore[override]
        self._hub = hub
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_request,
                condition=TypeCondition(DelegationRequest),
            )
        )
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_result,
                condition=TypeCondition(DelegationResult),
            )
        )

    def uninstall(self) -> None:
        if self._hub:
            for sub_id in self._sub_ids:
                self._hub.stream.unsubscribe(sub_id)
        self._sub_ids.clear()
        self._hub = None

    async def _on_request(self, event: DelegationRequest, ctx: Context) -> None:  # type: ignore[override]
        target = event.target
        self.specialist_activity[target] = self.specialist_activity.get(target, 0) + 1
        self.total_insights += 1
        self.delegation_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "request",
            "source": event.source,
            "target": target,
            "task_preview": event.task[:140],
        })

    async def _on_result(self, event: DelegationResult, ctx: Context) -> None:  # type: ignore[override]
        self.delegation_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "result",
            "source": event.source,
            "target": event.target,
            "result_preview": event.result[:140],
        })


# =====================================================================
# Tools: Collector — scans social channels for mentions
# =====================================================================


@tool
async def search_x(query: str, max_results: int = 10) -> str:
    """Search X (Twitter) for mentions matching the query keywords."""
    await asyncio.sleep(random.uniform(0.2, 0.5))
    query_terms = [t.lower().strip("#") for t in query.lower().split()]
    matches = []
    for m in _X_MENTIONS:
        text_lower = m["text"].lower()
        if any(term in text_lower for term in query_terms):
            matches.append(m)
    matches = matches[:max_results]
    if not matches:
        return f"Found 0 X mentions for '{query}'."

    lines = [f"Found {len(matches)} X mentions for '{query}':\n"]
    for m in matches:
        lines.append(f"  [{m['id']}] @{m['author']} ({m['followers']:,} followers)")
        lines.append(f'  "{m["text"]}"')
        lines.append(f"  Engagement: {m['likes']:,} likes, {m['retweets']:,} RTs")
        lines.append(f"  Posted: {m['timestamp']}")
        lines.append("")
    return "\n".join(lines)


@tool
async def search_reddit(query: str, subreddits: str = "smartwatches,gadgets,wearables", max_results: int = 10) -> str:
    """Search Reddit for posts matching the query across specified subreddits."""
    await asyncio.sleep(random.uniform(0.2, 0.5))
    query_terms = [t.lower() for t in query.lower().split()]
    target_subs = [s.strip().lower() for s in subreddits.split(",")]
    matches = []
    for p in _REDDIT_POSTS:
        sub = p["subreddit"].lower().replace("r/", "")
        text_lower = (p["title"] + " " + p["text"]).lower()
        sub_match = any(s in sub for s in target_subs) or not target_subs
        query_match = any(term in text_lower for term in query_terms)
        if sub_match and query_match:
            matches.append(p)
    matches = matches[:max_results]
    if not matches:
        return f"Found 0 Reddit posts for '{query}'."

    lines = [f"Found {len(matches)} Reddit posts for '{query}':\n"]
    for p in matches:
        lines.append(f"  [{p['id']}] {p['subreddit']} — u/{p['author']}")
        lines.append(f'  Title: "{p["title"]}"')
        lines.append(f'  "{p["text"][:200]}{"..." if len(p["text"]) > 200 else ""}"')
        lines.append(f"  Engagement: {p['upvotes']:,} upvotes, {p['comments']:,} comments")
        lines.append(f"  Posted: {p['timestamp']}")
        lines.append("")
    return "\n".join(lines)


@tool
async def search_reviews(product_name: str, platform: str = "all", min_rating: int = 1, max_rating: int = 5) -> str:
    """Search Google Reviews and App Store reviews for a product."""
    await asyncio.sleep(random.uniform(0.2, 0.5))
    query_terms = [t.lower() for t in product_name.lower().split()]
    matches = []
    for r in _REVIEWS:
        if platform != "all" and r["platform"] != platform:
            continue
        if not (min_rating <= r["rating"] <= max_rating):
            continue
        text_lower = (r["title"] + " " + r["text"]).lower()
        if any(term in text_lower for term in query_terms) or "novaband" in text_lower or "x1" in text_lower:
            matches.append(r)
    if not matches:
        return f"Found 0 reviews for '{product_name}'."

    lines = [f"Found {len(matches)} reviews for '{product_name}':\n"]
    for r in matches:
        stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
        lines.append(f"  [{r['id']}] {r['platform']} — {r['author']} {stars}")
        lines.append(f'  "{r["title"]}"')
        lines.append(f'  "{r["text"][:200]}{"..." if len(r["text"]) > 200 else ""}"')
        lines.append(f"  Posted: {r['timestamp']}")
        lines.append("")
    return "\n".join(lines)


@tool
async def search_news(query: str, days_back: int = 7) -> str:
    """Search news articles for recent coverage of a topic."""
    await asyncio.sleep(random.uniform(0.2, 0.5))
    query_terms = [t.lower() for t in query.lower().split()]
    matches = []
    for article in _NEWS_ARTICLES:
        text_lower = (article["headline"] + " " + article["snippet"]).lower()
        if any(term in text_lower for term in query_terms):
            matches.append(article)
    if not matches:
        return f"Found 0 news articles for '{query}'."

    lines = [f"Found {len(matches)} news articles for '{query}':\n"]
    for a in matches:
        lines.append(f"  [{a['id']}] {a['source']}")
        lines.append(f'  Headline: "{a["headline"]}"')
        lines.append(f'  "{a["snippet"][:250]}{"..." if len(a["snippet"]) > 250 else ""}"')
        lines.append(f"  Published: {a['timestamp']}")
        lines.append("")
    return "\n".join(lines)


COLLECTOR_TOOLS = [search_x, search_reddit, search_reviews, search_news]


# =====================================================================
# Tools: Analyst — classifies, scores, and routes mentions
# =====================================================================


@tool
async def analyze_mention(mention_text: str, source: str, author: str, reach: int = 0) -> str:
    """Analyze a customer mention: classify category, score sentiment, assess urgency.

    Returns structured analysis including category, sentiment_score (-1.0 to 1.0),
    urgency (low/medium/high/critical), and recommended routing target.
    """
    await asyncio.sleep(random.uniform(0.1, 0.3))
    text_lower = mention_text.lower()

    # Category classification (simulated AI analysis)
    if any(kw in text_lower for kw in ["overheat", "burn", "fire", "melt", "rash", "irritat", "blister"]):
        category = "safety"
    elif any(kw in text_lower for kw in ["lawsuit", "class action", "legal", "cpsc", "sue", "recall"]):
        category = "legal"
    elif any(kw in text_lower for kw in ["battery", "drain", "crack", "broken", "defect", "fail", "won't turn on"]):
        category = "defect"
    elif (
        any(kw in text_lower for kw in ["viral", "worst", "rant", "thread", "fail"])
        and reach > 50_000
        or any(kw in text_lower for kw in ["stock", "shares", "downgrad", "analyst"])
    ):
        category = "pr_risk"
    elif any(
        kw in text_lower for kw in ["compar", "vs", "versus", "apple watch", "fitbit", "samsung", "garmin"]
    ) or any(kw in text_lower for kw in ["price", "overpriced", "expensive", "cost", "cheaper"]):
        category = "competitor"
    elif any(kw in text_lower for kw in ["wish", "should add", "feature", "missing", "need"]):
        category = "feature_request"
    elif any(kw in text_lower for kw in ["love", "great", "best", "amazing", "excellent", "phenomenal"]):
        category = "praise"
    else:
        category = "general"

    # Sentiment scoring (simulated)
    positive_kw = ["love", "great", "best", "amazing", "excellent", "happy", "good", "accurate", "phenomenal"]
    negative_kw = [
        "worst",
        "terrible",
        "awful",
        "broken",
        "defect",
        "fail",
        "disappointed",
        "unacceptable",
        "burn",
        "rash",
        "melt",
        "avoid",
        "useless",
        "insane",
        "overpriced",
    ]
    pos_count = sum(1 for kw in positive_kw if kw in text_lower)
    neg_count = sum(1 for kw in negative_kw if kw in text_lower)
    sentiment_score = round((pos_count - neg_count) / (pos_count + neg_count), 2) if pos_count + neg_count > 0 else 0.0

    # Urgency assessment
    if category in ("safety", "legal") or category == "pr_risk" and reach > 100_000:
        urgency = "critical"
    elif category == "pr_risk" or (category == "defect" and reach > 10_000):
        urgency = "high"
    elif category == "defect":
        urgency = "medium"
    else:
        urgency = "low"

    # Routing recommendation
    route_map = {
        "safety": "escalation",
        "legal": "escalation",
        "defect": "product-inspector",
        "pr_risk": "pr-responder",
        "competitor": "market-intel",
        "praise": "none (log positive feedback)",
        "feature_request": "none (log for product roadmap)",
        "general": "none (no action needed)",
    }
    recommended_route = route_map.get(category, "analyst")

    return (
        f"Analysis Result:\n"
        f"  category: {category}\n"
        f"  sentiment_score: {sentiment_score}\n"
        f"  urgency: {urgency}\n"
        f"  reach: {reach:,} followers/subscribers\n"
        f"  recommended_route: {recommended_route}\n"
        f"  source: {source}\n"
        f"  author: {author}\n"
        f"  summary: {mention_text[:100]}{'...' if len(mention_text) > 100 else ''}"
    )


@tool
async def check_trending_topics(time_window_hours: int = 24) -> str:
    """Check what topics are currently trending in NovaBand X1 discussions."""
    await asyncio.sleep(random.uniform(0.1, 0.3))

    # Simulated trending analysis
    topics = [
        {"topic": "Battery drain", "volume": 1_247, "trend": "↑ +340% vs last week", "sentiment": "very negative"},
        {"topic": "Overheating/safety", "volume": 892, "trend": "↑ +520% vs last week", "sentiment": "very negative"},
        {"topic": "Skin irritation", "volume": 312, "trend": "↑ +180% vs last week", "sentiment": "negative"},
        {"topic": "GPS accuracy", "volume": 234, "trend": "→ stable", "sentiment": "positive"},
        {"topic": "Workout tracking", "volume": 189, "trend": "→ stable", "sentiment": "positive"},
        {"topic": "Price vs competitors", "volume": 156, "trend": "↑ +60% vs last week", "sentiment": "mixed"},
        {"topic": "Screen quality", "volume": 89, "trend": "↑ +45% vs last week", "sentiment": "negative"},
    ]

    lines = [f"Trending Topics (last {time_window_hours}h):\n"]
    for t in topics:
        lines.append(f"  • {t['topic']}: {t['volume']:,} mentions ({t['trend']}) — {t['sentiment']}")
    lines.append("")
    lines.append("Top concern: Battery drain and overheating safety issues dominate discussions.")
    lines.append("Positive signals: GPS accuracy and workout tracking continue to receive praise.")
    return "\n".join(lines)


@tool
async def create_routing_brief(
    category: str, mention_ids: str, summary: str, urgency: str, sentiment_score: float
) -> str:
    """Create a structured brief for routing to a specialist agent.

    Bundles classified mentions into a formatted brief with category,
    urgency, and actionable summary for the specialist team.
    """
    await asyncio.sleep(random.uniform(0.1, 0.2))
    brief_id = f"BRIEF-{random.randint(10000, 99999)}"

    return (
        f"Routing Brief {brief_id}:\n"
        f"  Category: {category}\n"
        f"  Urgency: {urgency}\n"
        f"  Sentiment: {sentiment_score}\n"
        f"  Mentions: {mention_ids}\n"
        f"  Summary: {summary}\n"
        f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Status: Ready for specialist review"
    )


ANALYST_TOOLS = [analyze_mention, check_trending_topics, create_routing_brief]


# =====================================================================
# Tools: Product Inspector — investigates defects and quality issues
# =====================================================================


@tool
async def lookup_known_issues(component: str, symptom: str = "") -> str:
    """Look up known product issues in the defect database by component and symptom."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    matches = []
    for defect_id, defect in _KNOWN_DEFECTS.items():
        comp_match = component.lower() in defect["component"].lower()
        symp_match = not symptom or any(kw in defect["title"].lower() for kw in symptom.lower().split())
        if comp_match or symp_match:
            matches.append((defect_id, defect))

    if not matches:
        return f"No known issues found for component='{component}', symptom='{symptom}'."

    lines = [f"Found {len(matches)} known issue(s):\n"]
    for did, d in matches:
        lines.append(f"  [{did}] {d['title']}")
        lines.append(f"    Component: {d['component']}")
        lines.append(f"    Severity: {d['severity']}")
        lines.append(f"    Status: {d['status']}")
        lines.append(f"    Reported count: {d['reported_count']:,}")
        lines.append(f"    Firmware affected: {d['firmware_affected']}")
        lines.append("")
    return "\n".join(lines)


@tool
async def create_defect_report(
    title: str, severity: str, component: str, description: str, mention_count: int = 1
) -> str:
    """Create a new defect report or update an existing one with customer mention data."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    report_id = f"DEF-{random.randint(100, 999)}"
    priority = (
        "P0 — Immediate"
        if severity == "critical"
        else "P1 — High"
        if severity == "high"
        else "P2 — Medium"
        if severity == "medium"
        else "P3 — Low"
    )

    return (
        f"Defect Report Created:\n"
        f"  Report ID: {report_id}\n"
        f"  Title: {title}\n"
        f"  Component: {component}\n"
        f"  Severity: {severity}\n"
        f"  Priority: {priority}\n"
        f"  Customer mentions: {mention_count}\n"
        f"  Description: {description[:200]}\n"
        f"  Status: Open — assigned to engineering\n"
        f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


@tool
async def recommend_action(defect_id: str, severity: str, affected_units_estimate: int = 0) -> str:
    """Recommend corrective action based on defect severity and scale."""
    await asyncio.sleep(random.uniform(0.1, 0.3))

    if severity == "critical" or affected_units_estimate > 1000:
        action = "PRODUCT RECALL ADVISORY"
        steps = [
            "1. Issue immediate stop-sale notice to retail partners",
            "2. Publish safety advisory on novatech.com and social channels",
            "3. Initiate voluntary recall program with full refund option",
            "4. Notify CPSC within 24 hours (required for safety defects)",
            "5. Establish dedicated customer hotline for affected units",
            f"6. Estimated cost: ${affected_units_estimate * 380:,} (units × avg replacement cost)",
        ]
    elif severity == "high":
        action = "FIRMWARE FIX + CUSTOMER ADVISORY"
        steps = [
            "1. Fast-track firmware fix (target: 72-hour release)",
            "2. Publish known issue advisory with workaround",
            "3. Offer affected customers extended warranty",
            "4. Monitor fix effectiveness via telemetry",
            f"5. Estimated affected units: {affected_units_estimate:,}",
        ]
    else:
        action = "STANDARD FIX CYCLE"
        steps = [
            "1. Include fix in next scheduled firmware release",
            "2. Update FAQ with workaround",
            "3. Monitor customer sentiment post-fix",
        ]

    return f"Recommended Action for {defect_id}:\n  Action: {action}\n  Steps:\n" + "\n".join(f"    {s}" for s in steps)


PRODUCT_INSPECTOR_TOOLS = [lookup_known_issues, create_defect_report, recommend_action]


# =====================================================================
# Tools: Market Intel — competitive analysis and market positioning
# =====================================================================


@tool
async def analyze_competitor(competitor_name: str, dimension: str = "overall") -> str:
    """Analyze a competitor across specified dimension (overall, pricing, features, sentiment)."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    comp = _COMPETITOR_DATA.get(competitor_name)
    if not comp:
        available = ", ".join(_COMPETITOR_DATA.keys())
        return f"Competitor '{competitor_name}' not found. Available: {available}"

    lines = [f"Competitor Analysis: {competitor_name}\n"]
    lines.append(f"  Price: ${comp['price']}")
    lines.append(f"  Battery Life: {comp['battery_life']}")
    lines.append(f"  Market Share: {comp['market_share']}")
    lines.append(f"  Overall Sentiment: {comp['sentiment']}")
    lines.append(f"  Strengths: {', '.join(comp['strengths'])}")
    lines.append(f"  Weaknesses: {', '.join(comp['weaknesses'])}")
    lines.append("")

    # NovaBand X1 comparison
    x1_price = 349
    price_diff = x1_price - comp["price"]
    if price_diff > 0:
        lines.append(f"  vs NovaBand X1: We are ${price_diff} MORE expensive")
        lines.append(f"  Price gap analysis: {competitor_name} offers better value at current pricing")
    elif price_diff < 0:
        lines.append(f"  vs NovaBand X1: We are ${abs(price_diff)} LESS expensive")
        lines.append("  Price gap analysis: NovaBand X1 positioned as value alternative")
    else:
        lines.append("  vs NovaBand X1: Price parity")

    return "\n".join(lines)


@tool
async def compare_features(our_product: str, competitor_product: str, feature_category: str = "all") -> str:
    """Compare features between NovaBand X1 and a competitor product."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    comp = _COMPETITOR_DATA.get(competitor_product)
    if not comp:
        return f"Competitor '{competitor_product}' not found."

    # Simulated feature comparison
    comparisons = {
        "Apple Watch Ultra 3": {
            "GPS accuracy": ("Excellent", "Excellent", "tie"),
            "Heart rate": ("Good", "Excellent", "lose"),
            "Battery life": ("Poor (18h)", "Good (48h)", "lose"),
            "Display": ("Excellent AMOLED", "Excellent OLED", "tie"),
            "App ecosystem": ("Limited", "Extensive", "lose"),
            "Price": ("$349", "$799", "win"),
            "Build quality": ("Good", "Excellent", "lose"),
        },
        "Samsung Galaxy Watch 7": {
            "GPS accuracy": ("Excellent", "Good", "win"),
            "Heart rate": ("Good", "Good", "tie"),
            "Battery life": ("Poor (18h)", "Good (40h)", "lose"),
            "Display": ("Excellent AMOLED", "Good AMOLED", "win"),
            "App ecosystem": ("Limited", "Moderate", "lose"),
            "Price": ("$349", "$279", "lose"),
            "Build quality": ("Good", "Good", "tie"),
        },
        "Fitbit Sense 4": {
            "GPS accuracy": ("Excellent", "Good", "win"),
            "Heart rate": ("Good", "Good", "tie"),
            "Battery life": ("Poor (18h)", "Excellent (6d)", "lose"),
            "Display": ("Excellent AMOLED", "Moderate LCD", "win"),
            "Health features": ("Basic", "Comprehensive", "lose"),
            "Price": ("$349", "$299", "lose"),
            "Build quality": ("Good", "Moderate", "win"),
        },
    }

    comp_data = comparisons.get(competitor_product, {})
    if not comp_data:
        return f"No feature comparison available for '{competitor_product}'."

    lines = [f"Feature Comparison: NovaBand X1 vs {competitor_product}\n"]
    wins = losses = ties = 0
    for feature, (ours, theirs, result) in comp_data.items():
        icon = "✓" if result == "win" else "✗" if result == "lose" else "="
        lines.append(f"  {icon} {feature}: {ours} vs {theirs}")
        if result == "win":
            wins += 1
        elif result == "lose":
            losses += 1
        else:
            ties += 1

    lines.append(f"\n  Summary: {wins} wins, {losses} losses, {ties} ties")
    if losses > wins:
        lines.append(f"  Assessment: {competitor_product} has competitive advantages we need to address")
    elif wins > losses:
        lines.append("  Assessment: NovaBand X1 holds competitive edge in key areas")
    else:
        lines.append("  Assessment: Products are closely matched — differentiation needed")

    return "\n".join(lines)


@tool
async def generate_competitive_brief(competitor: str, customer_mentions_summary: str) -> str:
    """Generate a competitive intelligence brief combining market data with customer mentions."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    brief_id = f"CI-{random.randint(1000, 9999)}"
    comp = _COMPETITOR_DATA.get(competitor, {})

    return (
        f"Competitive Intelligence Brief {brief_id}:\n"
        f"  Competitor: {competitor}\n"
        f"  Their Market Share: {comp.get('market_share', 'unknown')}\n"
        f"  Their Price Point: ${comp.get('price', 'N/A')}\n"
        f"  NovaBand X1 Price: $349\n\n"
        f"  Customer Voice Summary:\n"
        f"    {customer_mentions_summary[:300]}\n\n"
        f"  Key Takeaways:\n"
        f"    1. Customers frequently compare on price and battery life\n"
        f"    2. NovaBand X1 GPS accuracy is a cited advantage\n"
        f"    3. Battery life gap is the most cited competitive disadvantage\n\n"
        f"  Recommended Actions:\n"
        f"    1. Prioritize battery life improvement in next firmware/hardware revision\n"
        f"    2. Consider promotional pricing to close gap with {competitor}\n"
        f"    3. Amplify GPS accuracy advantage in marketing materials\n\n"
        f"  Brief ID: {brief_id}\n"
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


MARKET_INTEL_TOOLS = [analyze_competitor, compare_features, generate_competitive_brief]


# =====================================================================
# Tools: PR Responder — viral risk assessment and crisis communication
# =====================================================================


@tool
async def assess_viral_risk(platform: str, author_followers: int, engagement_rate: float, sentiment: str) -> str:
    """Assess the viral risk of a negative mention based on reach and engagement."""
    await asyncio.sleep(random.uniform(0.1, 0.3))

    # Risk scoring
    reach_score = min(author_followers / 100_000, 5.0)
    engagement_score = min(engagement_rate * 10, 5.0)
    sentiment_multiplier = 1.5 if sentiment in ("very_negative", "negative") else 1.0
    risk_score = round((reach_score + engagement_score) * sentiment_multiplier, 1)

    if risk_score >= 7.0:
        risk_level = "CRITICAL"
        response_window = "1 hour"
    elif risk_score >= 4.0:
        risk_level = "HIGH"
        response_window = "4 hours"
    elif risk_score >= 2.0:
        risk_level = "MODERATE"
        response_window = "24 hours"
    else:
        risk_level = "LOW"
        response_window = "48 hours"

    return (
        f"Viral Risk Assessment:\n"
        f"  Platform: {platform}\n"
        f"  Author Reach: {author_followers:,} followers\n"
        f"  Engagement Rate: {engagement_rate:.1%}\n"
        f"  Sentiment: {sentiment}\n"
        f"  Risk Score: {risk_score}/10\n"
        f"  Risk Level: {risk_level}\n"
        f"  Recommended Response Window: {response_window}\n"
        f"  Estimated Potential Impressions: {int(author_followers * engagement_rate * 10):,}"
    )


@tool
async def draft_response(platform: str, issue_summary: str, tone: str = "empathetic") -> str:
    """Draft a public response for a specific platform and issue."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    response_id = f"RESP-{random.randint(1000, 9999)}"

    # Simulated response drafts
    if "safety" in issue_summary.lower() or "burn" in issue_summary.lower() or "overheat" in issue_summary.lower():
        draft = (
            "We take every safety report seriously. We're aware of overheating reports "
            "with the NovaBand X1 and our engineering team is actively investigating. "
            "If you're experiencing this issue, please stop using the charging cradle "
            "and contact us at safety@novatech.com for an immediate replacement. "
            "Your safety is our top priority."
        )
    elif "battery" in issue_summary.lower():
        draft = (
            "We hear you on battery life — this isn't the experience we designed for. "
            "Our team has identified the root cause and a firmware fix is in final "
            "testing. We expect to push the update within the week. Thank you for "
            "your patience as we make this right."
        )
    else:
        draft = (
            "Thank you for sharing your experience. We're committed to making the "
            "NovaBand X1 the best it can be, and your feedback is invaluable. We're "
            "actively working on improvements and will keep the community updated. "
            "Please reach out to support@novatech.com for any immediate concerns."
        )

    return (
        f"Response Draft {response_id}:\n"
        f"  Platform: {platform}\n"
        f"  Tone: {tone}\n"
        f"  Issue: {issue_summary[:150]}\n\n"
        f"  Draft:\n"
        f'  "{draft}"\n\n'
        f"  Guidelines:\n"
        f"    - Respond on the same platform where the mention appeared\n"
        f"    - Use the author's name if available\n"
        f"    - Do NOT make promises about timelines unless confirmed by engineering\n"
        f"    - Escalate to legal if the mention references lawsuits or regulatory action"
    )


@tool
async def create_comms_plan(situation_summary: str, severity: str = "high") -> str:
    """Create a communications plan for managing a PR situation."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    plan_id = f"COMMS-{random.randint(1000, 9999)}"

    if severity == "critical":
        channels = [
            "CEO statement",
            "Press release",
            "Social media (all)",
            "Email to customers",
            "Retail partner notification",
            "Investor relations",
        ]
        timeline = "Immediate (within 2 hours)"
    elif severity == "high":
        channels = ["VP of Communications statement", "Social media response", "Blog post", "Customer support briefing"]
        timeline = "Within 4 hours"
    else:
        channels = ["Social media response", "FAQ update", "Support team briefing"]
        timeline = "Within 24 hours"

    return (
        f"Communications Plan {plan_id}:\n"
        f"  Severity: {severity}\n"
        f"  Timeline: {timeline}\n"
        f"  Situation: {situation_summary[:200]}\n\n"
        f"  Channels:\n" + "\n".join(f"    {i + 1}. {ch}" for i, ch in enumerate(channels)) + "\n\n  Key Messages:\n"
        "    1. Acknowledge the issue transparently\n"
        "    2. Outline concrete steps being taken\n"
        "    3. Provide direct support contact\n"
        "    4. Commit to follow-up communication with timeline\n\n"
        "  Stakeholder Notifications:\n"
        "    - Engineering: Immediate briefing on customer-facing issues\n"
        "    - Legal: Review all external communications before publishing\n"
        "    - Executive team: Situation brief within 1 hour\n"
        "    - Customer support: Updated talking points and escalation paths"
    )


PR_RESPONDER_TOOLS = [assess_viral_risk, draft_response, create_comms_plan]


# =====================================================================
# Tools: Escalation — urgent ticket creation and stakeholder alerts
# =====================================================================


@tool
async def create_urgent_ticket(title: str, severity: str, description: str, category: str = "safety") -> str:
    """Create an urgent ticket in the incident management system."""
    await asyncio.sleep(random.uniform(0.2, 0.4))
    ticket_id = f"URG-{random.randint(10000, 99999)}"

    sla = {
        "critical": "Response: 30 min, Resolution plan: 2 hours",
        "high": "Response: 1 hour, Resolution plan: 4 hours",
        "medium": "Response: 4 hours, Resolution plan: 24 hours",
    }

    return (
        f"Urgent Ticket Created:\n"
        f"  Ticket ID: {ticket_id}\n"
        f"  Title: {title}\n"
        f"  Category: {category}\n"
        f"  Severity: {severity}\n"
        f"  SLA: {sla.get(severity, 'Standard')}\n"
        f"  Description: {description[:250]}\n"
        f"  Status: Open — awaiting assignment\n"
        f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Escalation chain: On-call engineer -> Engineering Lead -> VP Engineering"
    )


@tool
async def notify_stakeholders(ticket_id: str, stakeholder_groups: str, message: str) -> str:
    """Send urgent notifications to specified stakeholder groups."""
    await asyncio.sleep(random.uniform(0.1, 0.3))
    groups = [g.strip() for g in stakeholder_groups.split(",")]

    notifications = []
    for group in groups:
        channel = {
            "engineering": "Slack #eng-oncall + PagerDuty",
            "legal": "Email to legal@novatech.com + Slack #legal-urgent",
            "executive": "SMS to C-suite + Slack #exec-alerts",
            "product": "Slack #product-safety + Email to PM leads",
            "customer_support": "Zendesk macro update + Slack #cs-alerts",
            "pr": "Slack #comms-war-room + SMS to VP Communications",
        }.get(group, f"Email to {group}")
        notifications.append(f"  ✓ {group}: notified via {channel}")

    return (
        f"Stakeholder Notifications for {ticket_id}:\n" + "\n".join(notifications) + f"\n\n  Message sent:\n"
        f'  "{message[:200]}"\n\n'
        f"  All notifications delivered at {datetime.now().strftime('%H:%M:%S')}"
    )


@tool
async def assess_legal_risk(issue_description: str, mention_count: int = 1, severity: str = "high") -> str:
    """Assess legal risk from customer complaints including class action potential."""
    await asyncio.sleep(random.uniform(0.2, 0.4))

    # Risk factors
    factors = []
    description_lower = issue_description.lower()
    if any(kw in description_lower for kw in ["burn", "fire", "melt", "injury"]):
        factors.append(("Physical harm reported", "HIGH"))
    if any(kw in description_lower for kw in ["class action", "lawsuit", "sue"]):
        factors.append(("Legal action mentioned by consumers", "HIGH"))
    if any(kw in description_lower for kw in ["cpsc", "recall", "regulatory"]):
        factors.append(("Regulatory involvement", "CRITICAL"))
    if mention_count > 50:
        factors.append((f"High complaint volume ({mention_count}+ mentions)", "HIGH"))
    if any(kw in description_lower for kw in ["warranty", "refund", "fraud"]):
        factors.append(("Consumer protection claims", "MEDIUM"))

    overall_risk = (
        "CRITICAL"
        if any(f[1] == "CRITICAL" for f in factors)
        else "HIGH"
        if any(f[1] == "HIGH" for f in factors)
        else "MEDIUM"
    )

    lines = [
        "Legal Risk Assessment:",
        f"  Overall Risk: {overall_risk}",
        f"  Complaint Volume: {mention_count} mentions",
        f"  Severity: {severity}",
        "",
        "  Risk Factors:",
    ]
    for factor, level in factors:
        lines.append(f"    [{level}] {factor}")

    lines.extend([
        "",
        "  Recommended Legal Actions:",
        "    1. Preserve all customer complaint records and social media mentions",
        "    2. Engage product liability counsel for preliminary assessment",
        "    3. Review product insurance coverage and notification obligations",
        "    4. Prepare regulatory response if CPSC complaint threshold is met",
        "    5. Document all corrective actions taken (for litigation defense)",
    ])

    return "\n".join(lines)


ESCALATION_TOOLS = [create_urgent_ticket, notify_stakeholders, assess_legal_risk]


# =====================================================================
# Agent creation
# =====================================================================


def create_collector(model: str, observers: list) -> Actor:
    return Actor(
        "collector",
        prompt=(
            "You are a Voice of Customer data collector for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to scan social media channels, review platforms, and news sources "
            "to collect all customer mentions about the NovaBand X1.\n\n"
            "Workflow:\n"
            "1. Search X (Twitter) for mentions using search_x with relevant keywords\n"
            "2. Search Reddit for discussions using search_reddit\n"
            "3. Search product reviews using search_reviews\n"
            "4. Search news coverage using search_news\n"
            "5. After collecting mentions from all available channels, compile a comprehensive "
            "summary of everything you found and delegate to the analyst agent using "
            "discover_agents and delegate_to. Include ALL mention IDs, the raw text, "
            "source platform, author info, and engagement metrics.\n\n"
            "Be thorough — scan ALL channels. Do not skip any source. The analyst needs "
            "complete data to classify and route effectively."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=COLLECTOR_TOOLS,
        observers=observers,
    )


def create_analyst(model: str, observers: list) -> Actor:
    return Actor(
        "analyst",
        prompt=(
            "You are a Voice of Customer analyst for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to classify customer mentions, assess sentiment and urgency, and "
            "route actionable insights to the right specialist team.\n\n"
            "Workflow:\n"
            "1. For each mention received, use analyze_mention to classify its category, "
            "score sentiment, and assess urgency\n"
            "2. Use check_trending_topics to understand the current landscape\n"
            "3. Group related mentions by category and create routing briefs using "
            "create_routing_brief\n"
            "4. Route briefs to specialist agents based on category:\n"
            "   - defect/quality issues -> delegate to product-inspector\n"
            "   - competitor/pricing mentions -> delegate to market-intel\n"
            "   - viral/PR-sensitive posts -> delegate to pr-responder\n"
            "   - safety/legal concerns -> delegate to escalation (HIGHEST PRIORITY)\n"
            "   - praise/feature requests -> log and summarize (no delegation needed)\n\n"
            "Use discover_agents to find available specialist agents, then delegate_to "
            "with the routing brief. ALWAYS prioritize safety and legal issues — route "
            "these first regardless of other findings.\n\n"
            "When delegating, include the full routing brief with mention details, "
            "category, urgency level, and actionable summary."
        ),
        config=GeminiConfig(
            model=model if "pro" in model else "gemini-3.1-pro-preview",
            temperature=0.2,
        ),
        tools=ANALYST_TOOLS,
        observers=observers,
    )


def create_product_inspector(model: str) -> Actor:
    return Actor(
        "product-inspector",
        prompt=(
            "You are a product quality inspector for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to investigate product defects reported by customers, cross-reference "
            "with known issues, and recommend corrective actions.\n\n"
            "Workflow:\n"
            "1. Review the incoming defect brief from the analyst\n"
            "2. Use lookup_known_issues to check if these defects are already tracked\n"
            "3. Create or update defect reports using create_defect_report with details "
            "from customer mentions\n"
            "4. Use recommend_action to propose corrective action based on severity "
            "and estimated impact\n\n"
            "For critical defects (especially safety-related), recommend the most "
            "protective action. For recurring issues with high report counts, escalate "
            "the recommended action level. Always include the estimated number of "
            "affected units in your analysis."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=PRODUCT_INSPECTOR_TOOLS,
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000)],
    )


def create_market_intel(model: str) -> Actor:
    return Actor(
        "market-intel",
        prompt=(
            "You are a competitive intelligence analyst for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to analyze competitor mentions, pricing discussions, and market "
            "positioning based on customer voice data.\n\n"
            "Workflow:\n"
            "1. Review the competitive intelligence brief from the analyst\n"
            "2. Use analyze_competitor to pull detailed data on mentioned competitors\n"
            "3. Use compare_features for direct product comparisons customers are making\n"
            "4. Use generate_competitive_brief to create an actionable intelligence report\n\n"
            "Focus on actionable insights: where are we losing to competitors in customer "
            "perception? What competitive advantages should we amplify? What pricing "
            "adjustments could improve market position?"
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=MARKET_INTEL_TOOLS,
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000)],
    )


def create_pr_responder(model: str) -> Actor:
    return Actor(
        "pr-responder",
        prompt=(
            "You are a PR and communications specialist for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to assess viral risk from negative mentions, draft appropriate "
            "responses, and create communications plans for PR-sensitive situations.\n\n"
            "Workflow:\n"
            "1. Review the PR brief from the analyst\n"
            "2. Use assess_viral_risk for high-reach negative mentions to quantify "
            "the risk and determine response urgency\n"
            "3. Use draft_response to create platform-appropriate responses\n"
            "4. For high/critical risk situations, use create_comms_plan to coordinate "
            "a multi-channel response strategy\n\n"
            "Key principles:\n"
            "- Respond quickly to high-reach negative mentions (within recommended window)\n"
            "- Never make promises about timelines unless confirmed by engineering\n"
            "- Escalate to legal review if mentions reference lawsuits or regulatory action\n"
            "- Tone should be empathetic, transparent, and action-oriented"
        ),
        config=GeminiConfig(
            model=model if "pro" in model else "gemini-3.1-pro-preview",
            temperature=0.3,
        ),
        tools=PR_RESPONDER_TOOLS,
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000)],
    )


def create_escalation(model: str) -> Actor:
    return Actor(
        "escalation",
        prompt=(
            "You are the escalation manager for NovaTech's NovaBand X1 smartwatch.\n\n"
            "Your job is to handle urgent safety, legal, and critical issues that require "
            "immediate attention and cross-functional coordination.\n\n"
            "Workflow:\n"
            "1. Review the escalation brief from the analyst — these are URGENT issues\n"
            "2. Use create_urgent_ticket to create an incident ticket with appropriate "
            "severity and SLA\n"
            "3. Use notify_stakeholders to alert relevant teams (engineering, legal, "
            "executive, product, customer support, PR)\n"
            "4. Use assess_legal_risk for any mentions involving lawsuits, regulatory "
            "complaints, or physical harm\n\n"
            "CRITICAL GUIDELINES:\n"
            "- Safety issues (burns, overheating, skin reactions) are ALWAYS critical severity\n"
            "- Legal mentions (class action, CPSC, lawsuits) require immediate legal notification\n"
            "- Physical harm reports require engineering + legal + executive notification\n"
            "- Document everything — every action must be traceable for potential litigation"
        ),
        config=GeminiConfig(
            model=model if "pro" in model else "gemini-3.1-pro-preview",
            temperature=0.1,
        ),
        tools=ESCALATION_TOOLS,
        observers=[
            TokenMonitor(warn_threshold=15_000, alert_threshold=40_000),
            LoopDetector(),
        ],
    )


# =====================================================================
# Scenarios
# =====================================================================

SCENARIOS = {
    1: (
        "Product Launch Monitoring — Full Channel Scan",
        "collector",
        (
            "NovaTech launched the NovaBand X1 smartwatch 3 days ago. Perform a comprehensive "
            "scan of all customer feedback channels: search X for #NovaBandX1 and NovaBand "
            "mentions, search Reddit (r/smartwatches, r/gadgets, r/wearables), search Google "
            "Reviews and App Store reviews for NovaBand, and search news for NovaTech NovaBand "
            "coverage. Collect ALL mentions and delegate the complete findings to the analyst "
            "for classification and routing to specialist teams."
        ),
    ),
    2: (
        "Crisis Detection — Safety Issue Spike",
        "collector",
        (
            "URGENT: We are receiving reports of the NovaBand X1 overheating during charging "
            "and causing skin burns during exercise. Perform an immediate scan focused on "
            "safety-related mentions: search X for NovaBand overheating burn safety, search "
            "Reddit for NovaBand safety charging issues, search reviews for NovaBand (filter "
            "1-2 star ratings), and search news for NovaBand safety investigation. Collect "
            "all safety-related mentions and delegate to the analyst with URGENT priority. "
            "The analyst should prioritize routing safety and legal issues to escalation."
        ),
    ),
    3: (
        "Autonomous Monitoring — Continuous VoC Pipeline",
        None,  # Scheduler-driven, no initial agent
        "",
    ),
    4: (
        "Competitive Intelligence — Market Positioning",
        "collector",
        (
            "Research how customers are comparing the NovaBand X1 to competitors. Search X "
            "for NovaBand vs Apple Watch, NovaBand vs Samsung, NovaBand vs Fitbit. Search "
            "Reddit for NovaBand comparison and competitor discussions. Search reviews and "
            "news for competitive coverage. Focus on collecting mentions where customers "
            "directly compare the NovaBand X1 to competing products. Delegate all competitive "
            "mentions to the analyst for classification and routing to the market intelligence team."
        ),
    ),
}


# =====================================================================
# Main
# =====================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice of Customer Intelligence Demo")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--duration", type=int, default=60, help="Scheduler duration in seconds (scenario 3)")
    parser.add_argument("message", nargs="?", default=None)
    args = parser.parse_args()

    scenario_title, entry_agent_name, message = SCENARIOS[args.scenario]
    if args.message:
        message = args.message
        scenario_title = "Custom Task"

    model = args.model

    # -- Create observers (keep references for end-of-run summary) -----
    volume_tracker = VolumeTracker()
    sentiment_monitor = SentimentMonitor()
    source_health_check = SourceHealthCheck()

    # -- Create agents ------------------------------------------------
    collector = create_collector(
        model,
        observers=[
            source_health_check,
            TokenMonitor(warn_threshold=15_000, alert_threshold=40_000),
        ],
    )
    analyst = create_analyst(
        model,
        observers=[
            volume_tracker,
            sentiment_monitor,
            TokenMonitor(warn_threshold=20_000, alert_threshold=50_000),
            LoopDetector(),
        ],
    )
    product_inspector = create_product_inspector(model)
    market_intel = create_market_intel(model)
    pr_responder = create_pr_responder(model)
    escalation = create_escalation(model)

    agent_map = {
        "collector": collector,
        "analyst": analyst,
        "product-inspector": product_inspector,
        "market-intel": market_intel,
        "pr-responder": pr_responder,
        "escalation": escalation,
    }

    # -- Create plugins -----------------------------------------------
    content_router = ContentRouter()
    insight_tracker = InsightTracker()
    telemetry = TelemetryPlugin()

    # -- Create network -----------------------------------------------
    network = Network(
        topology=Pipeline(content_router),
        plugins=[insight_tracker, telemetry],
        max_delegation_depth=6,
    )

    await network.register(
        collector,
        capabilities=["social-listening", "data-collection", "channel-scanning"],
        description="Collector — scans X, Reddit, reviews, and news for customer mentions",
    )
    await network.register(
        analyst,
        capabilities=["classification", "sentiment-analysis", "routing", "trend-analysis"],
        description="Analyst — classifies mentions by category, scores sentiment, routes to specialists",
    )
    await network.register(
        product_inspector,
        capabilities=["defect-investigation", "quality-analysis", "corrective-action"],
        description="Product Inspector — investigates defects and recommends corrective actions",
    )
    await network.register(
        market_intel,
        capabilities=["competitive-analysis", "market-positioning", "pricing-intelligence"],
        description="Market Intel — analyzes competitor mentions and market positioning",
    )
    await network.register(
        pr_responder,
        capabilities=["crisis-communication", "viral-risk", "response-drafting"],
        description="PR Responder — assesses viral risk and drafts crisis communications",
    )
    await network.register(
        escalation,
        capabilities=["incident-management", "legal-risk", "stakeholder-notification"],
        description="Escalation — handles urgent safety and legal issues with cross-functional coordination",
    )

    # -- Subscribe to hub stream for live logging ---------------------
    async def _on_hub_event(event: object) -> None:
        if isinstance(event, SchedulerTriggerFired):
            print()
            print(f"  {_DIM}{_ts()}{_RESET}  {_CYAN}{_BOLD}=== SCHEDULER: {event.target.upper()} triggered ==={_RESET}")
            task_preview = event.task[:140]
            print(f"  {_DIM}{' ' * 13}{_RESET}  {_CYAN}{task_preview}{_RESET}")
            print()
        elif isinstance(event, DelegationRequest):
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_MAGENTA}{_BOLD}DELEGATE{_RESET} "
                f"{_MAGENTA}{event.source} -> {event.target}{_RESET}"
            )
            task_preview = event.task[:140]
            print(f"  {_DIM}{' ' * 13}{_RESET}  {_DIM}{task_preview}{'...' if len(event.task) > 140 else ''}{_RESET}")
        elif isinstance(event, DelegationResult):
            preview = event.result[:140].replace("\n", " | ")
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_GREEN}{_BOLD}RESULT{_RESET}   "
                f"{_GREEN}{event.target} -> {event.source}{_RESET}"
            )
            print(f"  {_DIM}{' ' * 13}{_RESET}  {_DIM}{preview}{'...' if len(event.result) > 140 else ''}{_RESET}")
        elif isinstance(event, DelegationRejected):
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_RED}{_BOLD}REJECTED{_RESET} "
                f"{_RED}{event.source} -> {event.target}: {event.reason}{_RESET}"
            )
        elif isinstance(event, ObserverAlert):
            sev = event.severity.upper() if isinstance(event.severity, str) else str(event.severity)
            color = _severity_color(sev)
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{color}{_BOLD}ALERT [{sev}]{_RESET} "
                f"{color}({event.source}) {event.message[:120]}{_RESET}"
            )

    network.hub.stream.subscribe(_on_hub_event)

    # -- Print header -------------------------------------------------
    print()
    print(f"  {_BOLD}{'=' * 68}{_RESET}")
    print(f"  {_BOLD}VOICE OF CUSTOMER INTELLIGENCE — NovaBand X1{_RESET}")
    print(f"  {_BOLD}{'=' * 68}{_RESET}")
    print()
    print(f"  {_CYAN}Scenario:{_RESET}  {scenario_title}")
    print(f"  {_CYAN}Model:{_RESET}     {model}")
    print(f"  {_CYAN}Agents:{_RESET}    collector, analyst (pro), product-inspector,")
    print("             market-intel, pr-responder (pro), escalation (pro)")
    print()
    print(f"  {_BOLD}VoC Pipeline:{_RESET}")
    print("    collector -> analyst -> product-inspector")
    print("                        -> market-intel")
    print("                        -> pr-responder")
    print("                        -> escalation")
    print()
    print(f"  {_BOLD}Observers:{_RESET}")
    print(f"    {_YELLOW}VolumeTracker{_RESET}      spikes in defect/safety mentions")
    print(f"    {_YELLOW}SentimentMonitor{_RESET}   overall sentiment trend drift")
    print(f"    {_YELLOW}SourceHealthCheck{_RESET}  data source reliability")
    print(f"    {_DIM}TokenMonitor{_RESET}       token budget per agent")
    print(f"    {_DIM}LoopDetector{_RESET}       repetitive behavior prevention")
    print()
    print(f"  {_BOLD}Plugins:{_RESET}")
    print(f"    {_BLUE}ContentRouter{_RESET}      validates analyst -> specialist routing")
    print(f"    {_DIM}InsightTracker{_RESET}     tracks VoC insights flowing through network")
    print(f"    {_DIM}TelemetryPlugin{_RESET}    delegation metrics")
    print()

    if message:
        print(f"  {_BOLD}Task:{_RESET}")
        words = message.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 76:
                print(line)
                line = "    " + w
            else:
                line += " " + w if len(line) > 4 else w
        if len(line) > 4:
            print(line)
        print()

    print(f"  {_DIM}{'─' * 68}{_RESET}")
    print(f"  {_BOLD}Log legend:{_RESET}")
    print(f"    {_MAGENTA}DELEGATE{_RESET}         agent-to-agent delegation")
    print(f"    {_GREEN}RESULT{_RESET}           delegation completed")
    print(f"    {_CYAN}SCHEDULER{_RESET}        scheduled watch fired")
    print(f"    {_BLUE}CONTENT ROUTER{_RESET}   routing validation")
    print(f"    {_YELLOW}ALERT{_RESET}            observer signal")
    print(f"  {_DIM}{'─' * 68}{_RESET}")
    print()

    # -- Run ----------------------------------------------------------
    t0 = time.monotonic()

    if args.scenario == 3:
        # Autonomous VoC monitoring — scheduler-driven
        network.schedule(
            IntervalWatch(20),
            target="collector",
            task=(
                "Periodic VoC scan: Search all channels (X, Reddit, reviews, news) for "
                "new NovaBand X1 mentions. Collect everything you find and delegate the "
                "complete findings to the analyst for classification and routing."
            ),
        )
        network.schedule(
            IntervalWatch(35),
            target="analyst",
            task=(
                "Trend review: Use check_trending_topics to review the current mention "
                "landscape. Identify the top concerns and positive signals. If there are "
                "any emerging patterns that specialist teams should know about, create "
                "routing briefs and delegate to the appropriate specialists."
            ),
        )

        print(f"  {_BOLD}Starting autonomous VoC monitoring...{_RESET}")
        print(f"  {_DIM}(Scheduler running for {args.duration}s with 2 autonomous cycles){_RESET}")
        print("    Collector channel scan: every 20s")
        print("    Analyst trend review: every 35s")
        print()

        async with network:
            await asyncio.sleep(args.duration)

    else:
        # Interactive scenario
        entry_agent = agent_map[entry_agent_name]  # type: ignore[index]
        async with network:
            reply = await network.ask(entry_agent, message)

        elapsed = time.monotonic() - t0
        print()
        print(f"  {_BOLD}{'=' * 68}{_RESET}")
        print(f"  {_GREEN}{_BOLD}VoC PIPELINE RESULT{_RESET}  {_DIM}({elapsed:.1f}s){_RESET}")
        print(f"  {_BOLD}{'=' * 68}{_RESET}")
        print()
        for rline in (reply.body or "").split("\n"):
            print(f"  {rline}")
        print()

    # -- Print summary ------------------------------------------------
    elapsed_total = time.monotonic() - t0

    print(f"  {_DIM}{'─' * 68}{_RESET}")
    print(f"  {_BOLD}Content Router:{_RESET}")
    print(f"    Routes validated: {content_router.total_routed}")
    print(f"    Correct routes: {_GREEN}{content_router.correct_routes}{_RESET}")
    print(f"    Warnings: {_YELLOW}{content_router.total_warnings}{_RESET}")
    if content_router.route_log:
        print("    Route log:")
        for src, tgt, ok in content_router.route_log:
            status = f"{_GREEN}OK{_RESET}" if ok else f"{_YELLOW}WARN{_RESET}"
            print(f"      {src} -> {tgt}: {status}")
    print()

    print(f"  {_BOLD}Insight Tracker:{_RESET}")
    print(f"    Total insights routed: {insight_tracker.total_insights}")
    if insight_tracker.specialist_activity:
        print("    Specialist activity:")
        for agent, count in sorted(insight_tracker.specialist_activity.items(), key=lambda x: -x[1]):
            print(f"      {agent}: {count} delegation(s)")
    if insight_tracker.delegation_log:
        print("    Recent delegation flow:")
        for entry in insight_tracker.delegation_log[-8:]:
            etype = entry["type"]
            if etype == "request":
                print(f"    {_DIM}{entry['time']}{_RESET} {_MAGENTA}{entry['source']} -> {entry['target']}{_RESET}")
            else:
                print(f"    {_DIM}{entry['time']}{_RESET} {_GREEN}{entry['target']} completed{_RESET}")
    print()

    # Observer summaries
    print(f"  {_BOLD}Observer Reports:{_RESET}")

    # VolumeTracker (on analyst)
    s = volume_tracker.stats
    print("    VolumeTracker:")
    print(f"      Mentions analyzed: {s['total_analyzed']}")
    print(f"      Spikes detected: {s['spikes_detected']}")
    if s["category_counts"]:
        print(f"      Categories: {s['category_counts']}")

    # SentimentMonitor (on analyst)
    s = sentiment_monitor.stats
    print("    SentimentMonitor:")
    print(f"      Mentions scored: {s['total_scored']}")
    avg = s["average_sentiment"]
    color = _RED if avg < -0.3 else _YELLOW if avg < 0 else _GREEN
    print(f"      Average sentiment: {color}{avg}{_RESET}")
    print(f"      Warnings: {s['warnings']}")

    # SourceHealthCheck (on collector)
    s = source_health_check.stats
    print("    SourceHealthCheck:")
    for src_name, status in s["source_status"].items():
        src_color = _GREEN if status == "healthy" else _YELLOW
        print(f"      {src_name}: {src_color}{status}{_RESET}")
    if not s["source_status"]:
        print("      No sources monitored")
    print()

    print(f"  {_BOLD}Telemetry:{_RESET}")
    m = telemetry.metrics
    print(f"    Total delegations: {m.total_delegations}")
    print(f"    Total completions: {m.total_completions}")
    if m.by_target:
        print(f"    By target: {dict(m.by_target)}")
    print(f"    Total time: {elapsed_total:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
