import type { ReplayData } from '../replay-types'

export const replay11: ReplayData = {
  example: {
    id: '11',
    title: 'Voice of Customer',
    category: 'Content-Based Routing',
    description: 'Always-on social listening network. A collector scans channels, an analyst classifies with AI, and four specialists handle defects, competitive intel, PR crises, and safety escalations.',
    themes: ['network', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Product Launch Monitoring',
      inputMessage: 'NovaTech launched the NovaBand X1 smartwatch 3 days ago. Scan all customer feedback channels and route findings to specialist teams.',
      agents: [
        { id: 'collector', name: 'Collector', role: 'Social channel scanner', tools: ['search_x', 'search_reddit', 'search_reviews', 'search_news'], color: 'cyan', x: 50, y: 5 },
        { id: 'analyst', name: 'Analyst', role: 'AI classification & routing', tools: ['analyze_mention', 'check_trending', 'create_routing_brief'], color: 'blue', x: 50, y: 35 },
        { id: 'product-inspector', name: 'Product Inspector', role: 'Defect investigation', tools: ['lookup_known_issues', 'create_defect_report', 'recommend_action'], color: 'amber', x: 10, y: 75 },
        { id: 'market-intel', name: 'Market Intel', role: 'Competitive analysis', tools: ['analyze_competitor', 'compare_features', 'generate_brief'], color: 'emerald', x: 37, y: 75 },
        { id: 'pr-responder', name: 'PR Responder', role: 'Crisis communication', tools: ['assess_viral_risk', 'draft_response', 'create_comms_plan'], color: 'rose', x: 63, y: 75 },
        { id: 'escalation', name: 'Escalation', role: 'Safety & legal urgency', tools: ['create_urgent_ticket', 'notify_stakeholders', 'assess_legal_risk'], color: 'violet', x: 90, y: 75 },
      ],
      events: [
        // Collector scans all channels
        { id: 'e01', timestamp: 0, agent: 'collector', type: 'tool-call', toolName: 'search_x', args: 'query="NovaBand X1 #NovaBandX1"' },
        { id: 'e02', timestamp: 800, agent: 'collector', type: 'tool-result', toolName: 'search_x', result: 'Found 8 X mentions: battery complaints, safety warnings (overheating, skin rash), influencer viral rant (520K followers, 12K RTs), competitor comparisons, positive reviews.' },
        { id: 'e03', timestamp: 1500, agent: 'collector', type: 'tool-call', toolName: 'search_reddit', args: 'query="NovaBand X1", subreddits="smartwatches,gadgets,wearables"' },
        { id: 'e04', timestamp: 2300, agent: 'collector', type: 'tool-result', toolName: 'search_reddit', result: 'Found 6 Reddit posts: battery drain (847 upvotes), charger melting hazard (3,456 upvotes), class action discussion, competitor comparison, positive running review.' },
        { id: 'e05', timestamp: 3000, agent: 'collector', type: 'tool-call', toolName: 'search_reviews', args: 'product_name="NovaBand X1"' },
        { id: 'e06', timestamp: 3800, agent: 'collector', type: 'tool-result', toolName: 'search_reviews', result: 'Found 6 reviews: screen cracking (1★), great fitness tracker (5★), HR sensor inaccurate (2★), overpriced vs Samsung (4★), skin rash (1★), decent but lacking features (3★).' },
        { id: 'e07', timestamp: 4500, agent: 'collector', type: 'tool-call', toolName: 'search_news', args: 'query="NovaTech NovaBand X1"' },
        { id: 'e08', timestamp: 5300, agent: 'collector', type: 'tool-result', toolName: 'search_news', result: 'Found 4 articles: TechCrunch review (7/10), The Verge battery complaints, Consumer Reports safety investigation (47 CPSC complaints), Bloomberg stock drop 8.2%.' },

        // Collector delegates to analyst
        { id: 'e09', timestamp: 6500, agent: 'collector', type: 'discover-agents', capability: 'classification', results: ['analyst'] },
        { id: 'e10', timestamp: 7000, agent: 'collector', type: 'delegation-request', source: 'collector', target: 'analyst', channel: 'local', taskPreview: 'Classify and route 24 NovaBand X1 mentions from X (8), Reddit (6), Reviews (6), News (4). Includes safety concerns, defect reports, competitor comparisons, viral PR risk.' },

        // Analyst classifies mentions
        { id: 'e11', timestamp: 8000, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="SAFETY WARNING: NovaBand X1 got extremely hot during charging", source="x", author="@sarahk_running", reach=156000' },
        { id: 'e12', timestamp: 8800, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: safety, sentiment_score: -0.8, urgency: critical, recommended_route: escalation' },
        { id: 'e13', timestamp: 9500, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="Bought FIVE NovaBand X1 watches. THREE have screen defects", source="x", author="@viral_vinny", reach=520000' },
        { id: 'e14', timestamp: 10300, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: pr_risk, sentiment_score: -1.0, urgency: critical, recommended_route: pr-responder' },
        { id: 'e15', timestamp: 11000, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="Battery dying after 6 hours, supposed to last 3 days", source="x", author="@techreviewer_mike", reach=12400' },
        { id: 'e16', timestamp: 11800, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: defect, sentiment_score: -0.5, urgency: high, recommended_route: product-inspector' },
        { id: 'e17', timestamp: 12500, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="NovaBand X1 side-by-side with Apple Watch Ultra 3. GPS comparable but HR lags", source="x", author="@gadget_guru_sam", reach=45000' },
        { id: 'e18', timestamp: 13300, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: competitor, sentiment_score: 0.0, urgency: medium, recommended_route: market-intel' },
        { id: 'e19', timestamp: 14000, agent: 'analyst', type: 'tool-call', toolName: 'check_trending', args: 'time_window_hours=24' },
        { id: 'e20', timestamp: 14800, agent: 'analyst', type: 'tool-result', toolName: 'check_trending', result: 'Top concerns: Battery drain (1,247 mentions, +340%), Overheating/safety (892, +520%), Skin irritation (312, +180%). Positive: GPS accuracy, workout tracking.' },

        // Analyst creates briefs and routes to all 4 specialists in parallel
        { id: 'e21', timestamp: 16000, agent: 'analyst', type: 'tool-call', toolName: 'create_routing_brief', args: 'category="safety/legal", urgency="critical", mentions="x-1004,x-1007,reddit-2004,reddit-2005,review-3005"' },
        { id: 'e22', timestamp: 16500, agent: 'analyst', type: 'tool-result', toolName: 'create_routing_brief', result: 'Brief BRIEF-39191: Safety/Legal, critical urgency, 5 mentions covering overheating, charger melting, skin burns, class action discussion.' },
        { id: 'e23', timestamp: 17000, agent: 'analyst', type: 'discover-agents', capability: 'incident-management', results: ['escalation'] },

        // Parallel delegations to all 4 specialists
        { id: 'e24', timestamp: 17500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'escalation', channel: 'local', taskPreview: 'CRITICAL — Safety/legal brief: overheating during charging, charger melting, skin burns (47 CPSC complaints), class action discussion. Create urgent ticket and notify stakeholders.' },
        { id: 'e25', timestamp: 17500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'product-inspector', channel: 'local', taskPreview: 'Defect brief: battery drain (15%/hr vs claimed 72hr), screen cracking, HR sensor inaccuracy. Investigate known issues and recommend corrective actions.' },
        { id: 'e26', timestamp: 17500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'market-intel', channel: 'local', taskPreview: 'Competitor brief: customers comparing NovaBand X1 to Apple Watch Ultra 3, Samsung Galaxy Watch 7, Fitbit Sense 4. Pricing concerns ($349 vs $279 Samsung).' },
        { id: 'e27', timestamp: 17500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'pr-responder', channel: 'local', taskPreview: 'PR risk brief: viral thread from @viral_vinny (520K followers, 12K RTs), Bloomberg stock drop coverage, NovaTechFail hashtag trending.' },

        // Escalation handles safety
        { id: 'e28', timestamp: 19000, agent: 'escalation', type: 'tool-call', toolName: 'create_urgent_ticket', args: 'title="NovaBand X1 Overheating & Safety", severity="critical", category="safety"' },
        { id: 'e29', timestamp: 19800, agent: 'escalation', type: 'tool-result', toolName: 'create_urgent_ticket', result: 'Urgent ticket URG-48835 created. Severity: critical. SLA: Response 30 min, Resolution plan 2 hours.' },
        { id: 'e30', timestamp: 20500, agent: 'escalation', type: 'tool-call', toolName: 'notify_stakeholders', args: 'ticket_id="URG-48835", groups="engineering,legal,executive,product"' },
        { id: 'e31', timestamp: 21300, agent: 'escalation', type: 'tool-result', toolName: 'notify_stakeholders', result: '✓ engineering: Slack #eng-oncall + PagerDuty. ✓ legal: email + Slack #legal-urgent. ✓ executive: SMS + Slack #exec-alerts. ✓ product: Slack #product-safety.' },
        { id: 'e32', timestamp: 22000, agent: 'escalation', type: 'tool-call', toolName: 'assess_legal_risk', args: 'issue="charger melting, skin burns, 47 CPSC complaints, class action discussion", mention_count=47' },
        { id: 'e33', timestamp: 22800, agent: 'escalation', type: 'tool-result', toolName: 'assess_legal_risk', result: 'Legal risk: CRITICAL. Factors: [HIGH] Physical harm reported, [HIGH] Legal action mentioned, [CRITICAL] Regulatory involvement (CPSC). Recommend: preserve records, engage product liability counsel.' },

        // Product Inspector investigates defects
        { id: 'e34', timestamp: 19500, agent: 'product-inspector', type: 'tool-call', toolName: 'lookup_known_issues', args: 'component="battery", symptom="excessive drain"' },
        { id: 'e35', timestamp: 20300, agent: 'product-inspector', type: 'tool-result', toolName: 'lookup_known_issues', result: 'DEF-001: Excessive battery drain. Severity: high. Status: investigating. 1,247 reports. Firmware v1.0.0-v1.0.2.' },
        { id: 'e36', timestamp: 21000, agent: 'product-inspector', type: 'tool-call', toolName: 'create_defect_report', args: 'title="Battery drain customer VoC cluster", severity="high", component="battery"' },
        { id: 'e37', timestamp: 21800, agent: 'product-inspector', type: 'tool-result', toolName: 'create_defect_report', result: 'Defect Report DEF-247 created. Priority P1-High. Linked to DEF-001. Customer mentions: 4.' },
        { id: 'e38', timestamp: 22500, agent: 'product-inspector', type: 'tool-call', toolName: 'recommend_action', args: 'defect_id="DEF-001", severity="high", affected_units_estimate=1247' },
        { id: 'e39', timestamp: 23300, agent: 'product-inspector', type: 'tool-result', toolName: 'recommend_action', result: 'Recommended: FIRMWARE FIX + CUSTOMER ADVISORY. Fast-track 72-hour firmware fix, publish known issue advisory, offer extended warranty to affected customers.' },

        // Market Intel analyzes competitors
        { id: 'e40', timestamp: 20000, agent: 'market-intel', type: 'tool-call', toolName: 'analyze_competitor', args: 'competitor_name="Samsung Galaxy Watch 7", dimension="overall"' },
        { id: 'e41', timestamp: 20800, agent: 'market-intel', type: 'tool-result', toolName: 'analyze_competitor', result: 'Samsung Galaxy Watch 7: $279, 40hr battery, 18% share. We are $70 MORE expensive. Strengths: Android integration, price/value, battery.' },
        { id: 'e42', timestamp: 21500, agent: 'market-intel', type: 'tool-call', toolName: 'compare_features', args: 'our_product="NovaBand X1", competitor_product="Apple Watch Ultra 3"' },
        { id: 'e43', timestamp: 22300, agent: 'market-intel', type: 'tool-result', toolName: 'compare_features', result: 'NovaBand X1 vs Apple Watch Ultra 3: 2 wins (display, price), 4 losses (HR, battery, apps, build), 1 tie (GPS). Apple holds competitive edge.' },
        { id: 'e44', timestamp: 23000, agent: 'market-intel', type: 'tool-call', toolName: 'generate_brief', args: 'competitor="Samsung Galaxy Watch 7", findings="customers cite $70 price gap and battery gap"' },
        { id: 'e45', timestamp: 23800, agent: 'market-intel', type: 'tool-result', toolName: 'generate_brief', result: 'Brief CI-4821: Recommend $50 price reduction, amplify GPS advantage in marketing, prioritize battery fix to close competitive gap with Samsung.' },

        // PR Responder handles viral risk
        { id: 'e46', timestamp: 19800, agent: 'pr-responder', type: 'tool-call', toolName: 'assess_viral_risk', args: 'platform="x", author_followers=520000, engagement_rate=0.024, sentiment="very_negative"' },
        { id: 'e47', timestamp: 20600, agent: 'pr-responder', type: 'tool-result', toolName: 'assess_viral_risk', result: 'Risk Score: 11.1/10 CRITICAL. Response window: 1 hour. Estimated impressions: 124,800,000. Viral thread still growing.' },
        { id: 'e48', timestamp: 21300, agent: 'pr-responder', type: 'tool-call', toolName: 'draft_response', args: 'platform="x", issue_summary="quality defects — screen, power, viral rant", tone="empathetic"' },
        { id: 'e49', timestamp: 22100, agent: 'pr-responder', type: 'tool-result', toolName: 'draft_response', result: 'Draft RESP-7234: "Thank you for sharing your experience. We\'re committed to making the NovaBand X1 the best it can be..." Platform-appropriate, empathetic tone.' },
        { id: 'e50', timestamp: 22800, agent: 'pr-responder', type: 'tool-call', toolName: 'create_comms_plan', args: 'situation_summary="Viral quality complaints + stock drop + safety investigation", severity="critical"' },
        { id: 'e51', timestamp: 23600, agent: 'pr-responder', type: 'tool-result', toolName: 'create_comms_plan', result: 'Plan COMMS-5892: CEO statement + press release + social response + customer email + retail notification + investor relations. Timeline: immediate (2 hours).' },

        // Results flow back to analyst
        { id: 'e52', timestamp: 24000, agent: 'escalation', type: 'delegation-result', source: 'analyst', target: 'escalation', resultPreview: 'Urgent ticket URG-48835 created. Legal risk CRITICAL. Engineering, legal, executive, product teams notified. CPSC response required within 24 hours.' },
        { id: 'e53', timestamp: 24500, agent: 'product-inspector', type: 'delegation-result', source: 'analyst', target: 'product-inspector', resultPreview: 'Battery DEF-001 confirmed (1,247 reports). Recommended firmware fix in 72 hours + customer advisory. Screen cracking and HR sensor also tracked.' },
        { id: 'e54', timestamp: 25000, agent: 'market-intel', type: 'delegation-result', source: 'analyst', target: 'market-intel', resultPreview: 'Samsung $70 cheaper with better battery. Apple dominates on ecosystem. Recommend: $50 price cut, amplify GPS advantage, prioritize battery fix.' },
        { id: 'e55', timestamp: 25500, agent: 'pr-responder', type: 'delegation-result', source: 'analyst', target: 'pr-responder', resultPreview: 'Viral risk CRITICAL (11.1/10). Response within 1 hour required. CEO statement + press release + social response drafted. Full comms plan COMMS-5892 activated.' },

        // Analyst synthesizes and returns to collector
        { id: 'e56', timestamp: 27000, agent: 'analyst', type: 'model-response', contentPreview: 'VoC analysis complete. 24 mentions classified: 5 safety/legal (CRITICAL → escalation), 4 defects (HIGH → product-inspector), 5 competitor (MEDIUM → market-intel), 3 viral/PR (CRITICAL → pr-responder), 4 praise, 3 feature requests.' },
        { id: 'e57', timestamp: 28000, agent: 'analyst', type: 'delegation-result', source: 'collector', target: 'analyst', resultPreview: 'Full VoC pipeline complete. Safety: ticket URG-48835 + stakeholder alert. Defects: firmware fix in 72h. Competitive: $50 price cut recommended. PR: crisis comms plan activated.' },
        { id: 'e58', timestamp: 29000, agent: 'collector', type: 'model-response', contentPreview: 'Comprehensive VoC scan complete across 4 channels (24 mentions). Routed to all specialist teams. Critical actions: safety escalation (overheating), PR crisis response (viral thread), battery firmware fix, competitive pricing adjustment.' },
      ],
      totalDurationMs: 30000,
    },
    {
      id: 2,
      title: 'Crisis Detection — Safety Spike',
      inputMessage: 'URGENT: Reports of NovaBand X1 overheating and causing burns. Scan for safety-related mentions and escalate immediately.',
      agents: [
        { id: 'collector', name: 'Collector', role: 'Social channel scanner', tools: ['search_x', 'search_reddit', 'search_reviews', 'search_news'], color: 'cyan', x: 50, y: 5 },
        { id: 'analyst', name: 'Analyst', role: 'AI classification & routing', tools: ['analyze_mention', 'check_trending', 'create_routing_brief'], color: 'blue', x: 50, y: 35 },
        { id: 'product-inspector', name: 'Product Inspector', role: 'Defect investigation', tools: ['lookup_known_issues', 'create_defect_report', 'recommend_action'], color: 'amber', x: 25, y: 75 },
        { id: 'escalation', name: 'Escalation', role: 'Safety & legal urgency', tools: ['create_urgent_ticket', 'notify_stakeholders', 'assess_legal_risk'], color: 'violet', x: 75, y: 75 },
      ],
      events: [
        { id: 'c01', timestamp: 0, agent: 'collector', type: 'tool-call', toolName: 'search_x', args: 'query="NovaBand overheating burn safety"' },
        { id: 'c02', timestamp: 800, agent: 'collector', type: 'tool-result', toolName: 'search_x', result: 'Found 3 safety-related X mentions: overheating during charging (156K followers), skin rash reports, influencer warning.' },
        { id: 'c03', timestamp: 1500, agent: 'collector', type: 'tool-call', toolName: 'search_reddit', args: 'query="NovaBand safety charging burn"' },
        { id: 'c04', timestamp: 2300, agent: 'collector', type: 'tool-result', toolName: 'search_reddit', result: 'Found 2 critical posts: charger melted overnight (3,456 upvotes), class action for burn injury (2,109 upvotes).' },
        { id: 'c05', timestamp: 3000, agent: 'collector', type: 'tool-call', toolName: 'search_reviews', args: 'product_name="NovaBand", min_rating=1, max_rating=2' },
        { id: 'c06', timestamp: 3800, agent: 'collector', type: 'tool-result', toolName: 'search_reviews', result: 'Found 3 low-rated reviews: screen cracking, skin rash/blistering (dermatologist-confirmed contact dermatitis), sensor inaccuracy.' },
        { id: 'c07', timestamp: 4500, agent: 'collector', type: 'tool-call', toolName: 'search_news', args: 'query="NovaBand safety investigation"' },
        { id: 'c08', timestamp: 5300, agent: 'collector', type: 'tool-result', toolName: 'search_news', result: 'Consumer Reports launched safety investigation. 47 CPSC complaints filed. Bloomberg: stock down 8.2%.' },
        { id: 'c09', timestamp: 6000, agent: 'collector', type: 'delegation-request', source: 'collector', target: 'analyst', channel: 'local', taskPreview: 'URGENT safety scan: 8+ safety-related mentions across X, Reddit, reviews, news. Overheating, burns, charger melting, CPSC investigation. Prioritize escalation routing.' },

        // Analyst classifies — focused on safety
        { id: 'c10', timestamp: 7000, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="charger partially melted, scorch mark, fire hazard, CPSC complaint filed", source="reddit", reach=3456' },
        { id: 'c11', timestamp: 7800, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: safety, sentiment_score: -1.0, urgency: critical, recommended_route: escalation' },
        { id: 'c12', timestamp: 8500, agent: 'analyst', type: 'tool-call', toolName: 'analyze_mention', args: 'mention="first-degree burn on wrist, medical documentation, 30 others reporting, class action", source="reddit", reach=2109' },
        { id: 'c13', timestamp: 9300, agent: 'analyst', type: 'tool-result', toolName: 'analyze_mention', result: 'category: legal, sentiment_score: -1.0, urgency: critical, recommended_route: escalation' },

        // Routes to escalation and product-inspector
        { id: 'c14', timestamp: 10500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'escalation', channel: 'local', taskPreview: 'CRITICAL safety/legal escalation: charger melting, skin burns, 47 CPSC complaints, class action potential. Physical harm documented. Immediate action required.' },
        { id: 'c15', timestamp: 10500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'product-inspector', channel: 'local', taskPreview: 'Safety defect investigation: charging cradle overheating (DEF-003), sensor housing causing contact dermatitis (DEF-004). Recommend recall assessment.' },

        // Escalation handles crisis
        { id: 'c16', timestamp: 12000, agent: 'escalation', type: 'tool-call', toolName: 'create_urgent_ticket', args: 'title="SAFETY: NovaBand X1 Overheating/Burns", severity="critical"' },
        { id: 'c17', timestamp: 12800, agent: 'escalation', type: 'tool-result', toolName: 'create_urgent_ticket', result: 'URG-51002 created. Critical severity. SLA: 30 min response.' },
        { id: 'c18', timestamp: 13500, agent: 'escalation', type: 'tool-call', toolName: 'notify_stakeholders', args: 'ticket_id="URG-51002", groups="engineering,legal,executive,customer_support,pr"' },
        { id: 'c19', timestamp: 14300, agent: 'escalation', type: 'tool-result', toolName: 'notify_stakeholders', result: 'All 5 stakeholder groups notified. Engineering via PagerDuty, Legal via urgent email, Executive via SMS, CS via Zendesk, PR via war room.' },
        { id: 'c20', timestamp: 15000, agent: 'escalation', type: 'tool-call', toolName: 'assess_legal_risk', args: 'issue="burns, melting charger, 47 CPSC complaints, class action mentioned"' },
        { id: 'c21', timestamp: 15800, agent: 'escalation', type: 'tool-result', toolName: 'assess_legal_risk', result: 'Legal risk: CRITICAL. Physical harm + regulatory involvement + class action threat. Immediate counsel engagement required.' },

        // Product inspector investigates
        { id: 'c22', timestamp: 12500, agent: 'product-inspector', type: 'tool-call', toolName: 'lookup_known_issues', args: 'component="charging_cradle", symptom="overheating melting"' },
        { id: 'c23', timestamp: 13300, agent: 'product-inspector', type: 'tool-result', toolName: 'lookup_known_issues', result: 'DEF-003: Charging cradle overheating/melting. CRITICAL. Active investigation. 47 reports.' },
        { id: 'c24', timestamp: 14000, agent: 'product-inspector', type: 'tool-call', toolName: 'recommend_action', args: 'defect_id="DEF-003", severity="critical", affected_units_estimate=5000' },
        { id: 'c25', timestamp: 14800, agent: 'product-inspector', type: 'tool-result', toolName: 'recommend_action', result: 'PRODUCT RECALL ADVISORY: Stop-sale notice, CPSC notification within 24h, voluntary recall with full refund. Estimated cost: $1,900,000.' },

        // Results return
        { id: 'c26', timestamp: 16500, agent: 'escalation', type: 'delegation-result', source: 'analyst', target: 'escalation', resultPreview: 'Ticket URG-51002 created. All teams notified. Legal risk CRITICAL. CPSC notification required. Counsel engaged.' },
        { id: 'c27', timestamp: 17000, agent: 'product-inspector', type: 'delegation-result', source: 'analyst', target: 'product-inspector', resultPreview: 'RECALL ADVISORY for DEF-003 (charger). Stop-sale + CPSC notification + voluntary recall. Est. cost: $1.9M.' },
        { id: 'c28', timestamp: 18000, agent: 'analyst', type: 'delegation-result', source: 'collector', target: 'analyst', resultPreview: 'Crisis response activated. Safety escalation ticket + recall advisory + full stakeholder notification. All critical items handled.' },
        { id: 'c29', timestamp: 19000, agent: 'collector', type: 'model-response', contentPreview: 'URGENT VoC crisis scan complete. Safety escalation: ticket URG-51002 (critical), all stakeholders notified, product recall advisory issued for charging cradle (est. $1.9M). Legal counsel engaged for class action defense.' },
      ],
      totalDurationMs: 20000,
    },
  ],
}
