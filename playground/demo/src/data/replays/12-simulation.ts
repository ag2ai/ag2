import type { ReplayData } from '../replay-types'

export const replay12: ReplayData = {
  example: {
    id: '12',
    title: 'Decision Simulation',
    category: 'Many-Actor Network',
    description: 'Scale-out simulation: 11 actors analyze a business decision from 9 independent perspectives, then an analyst synthesizes the results.',
    themes: ['network', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'SaaS Price Increase',
      inputMessage:
        'We are considering raising our SaaS subscription prices by 25% across all tiers. Our product is a B2B analytics platform with 2,000 enterprise customers, $50M ARR, and 90% gross margins. Competitors have not raised prices recently.',
      agents: [
        { id: 'coordinator', name: 'Coordinator', role: 'Simulation coordinator', tools: ['structure_decision', 'discover_agents', 'delegate_to'], color: 'blue', x: 8, y: 50 },
        { id: 'cfo', name: 'CFO', role: 'Financial analysis', tools: ['run_financial_model'], color: 'emerald', x: 32, y: 12 },
        { id: 'cmo', name: 'CMO', role: 'Marketing & brand', tools: ['analyze_brand_impact'], color: 'cyan', x: 50, y: 12 },
        { id: 'cto', name: 'CTO', role: 'Technical feasibility', tools: ['assess_technical_feasibility'], color: 'violet', x: 68, y: 12 },
        { id: 'hr-director', name: 'HR Director', role: 'Workforce impact', tools: ['assess_workforce_impact'], color: 'amber', x: 32, y: 50 },
        { id: 'legal-counsel', name: 'Legal', role: 'Regulatory compliance', tools: ['check_regulatory_compliance'], color: 'rose', x: 50, y: 50 },
        { id: 'customer-advocate', name: 'Customer', role: 'Customer impact', tools: ['predict_customer_impact'], color: 'yellow', x: 68, y: 50 },
        { id: 'operations-lead', name: 'Operations', role: 'Supply chain & ops', tools: ['analyze_supply_chain_impact'], color: 'emerald', x: 32, y: 88 },
        { id: 'competitor-analyst', name: 'Competitor', role: 'Competitive response', tools: ['predict_competitor_response'], color: 'cyan', x: 50, y: 88 },
        { id: 'board-member', name: 'Board', role: 'Strategic governance', tools: ['evaluate_strategic_alignment'], color: 'amber', x: 68, y: 88 },
        { id: 'analyst', name: 'Analyst', role: 'Synthesis & recommendation', tools: ['synthesize_perspectives', 'generate_recommendation'], color: 'rose', x: 92, y: 50 },
      ],
      events: [
        // Coordinator structures the decision
        { id: 'e01', timestamp: 0, agent: 'coordinator', type: 'tool-call', toolName: 'structure_decision', args: 'decision="Raise SaaS prices 25%", time_horizon="12 months"' },
        { id: 'e02', timestamp: 800, agent: 'coordinator', type: 'tool-result', toolName: 'structure_decision', result: 'SIMULATION BRIEF: Raise SaaS prices 25% across all tiers. SIM-20260324-4821. Status: READY FOR DISTRIBUTION.' },

        // Coordinator discovers all simulation personas
        { id: 'e03', timestamp: 1500, agent: 'coordinator', type: 'discover-agents', capability: 'simulation', results: ['cfo', 'cmo', 'cto', 'hr-director', 'legal-counsel', 'customer-advocate', 'operations-lead', 'competitor-analyst', 'board-member'] },

        // Fan-out: delegate to all 9 personas simultaneously
        { id: 'e04', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cfo', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from financial perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e05', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cmo', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from marketing/brand perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e06', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cto', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from technical feasibility perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e07', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'hr-director', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from workforce/talent perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e08', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'legal-counsel', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from legal/regulatory perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e09', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'customer-advocate', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from customer satisfaction perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e10', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'operations-lead', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from operations perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e11', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'competitor-analyst', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from competitive landscape perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'e12', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'board-member', channel: 'local', taskPreview: 'Analyze 25% SaaS price increase from strategic governance perspective. State SUPPORT/OPPOSE/NEUTRAL.' },

        // Personas run their tools (staggered)
        { id: 'e13', timestamp: 3000, agent: 'cfo', type: 'tool-call', toolName: 'run_financial_model', args: 'revenue_impact_pct=+20, cost_impact_pct=+5, payback_months=3' },
        { id: 'e14', timestamp: 3200, agent: 'cmo', type: 'tool-call', toolName: 'analyze_brand_impact', args: 'brand_alignment_score=4, market_perception_risk="high"' },
        { id: 'e15', timestamp: 3500, agent: 'cto', type: 'tool-call', toolName: 'assess_technical_feasibility', args: 'engineering_months=2, infrastructure_changes="billing system update"' },
        { id: 'e16', timestamp: 3800, agent: 'hr-director', type: 'tool-call', toolName: 'assess_workforce_impact', args: 'headcount_change=+3, reskilling_needed=false, morale_risk="high"' },
        { id: 'e17', timestamp: 4000, agent: 'legal-counsel', type: 'tool-call', toolName: 'check_regulatory_compliance', args: 'jurisdictions="US, EU", data_privacy_impact=false' },
        { id: 'e18', timestamp: 4200, agent: 'customer-advocate', type: 'tool-call', toolName: 'predict_customer_impact', args: 'affected_segment="Enterprise", satisfaction_change=-20, churn_risk_pct=15' },
        { id: 'e19', timestamp: 4500, agent: 'operations-lead', type: 'tool-call', toolName: 'analyze_supply_chain_impact', args: 'vendor_changes="None", lead_time_weeks=4' },
        { id: 'e20', timestamp: 4800, agent: 'competitor-analyst', type: 'tool-call', toolName: 'predict_competitor_response', args: 'primary_competitors="Looker, Tableau", market_share_impact_pct=-5' },
        { id: 'e21', timestamp: 5000, agent: 'board-member', type: 'tool-call', toolName: 'evaluate_strategic_alignment', args: 'vision_alignment_score=5, investor_sentiment="neutral"' },

        // Tool results return
        { id: 'e22', timestamp: 5500, agent: 'cfo', type: 'tool-result', toolName: 'run_financial_model', result: 'Projected ARR: $60M (+20%). Cost Delta: +5%. Payback: 3mo. NPV (3yr): $22.5M. Risk: MODERATE.' },
        { id: 'e23', timestamp: 5800, agent: 'cmo', type: 'tool-result', toolName: 'analyze_brand_impact', result: 'Brand Alignment: 4/10. Perception Risk: HIGH. Competitor Narrative Risk: MODERATE. Enhanced crisis readiness needed.' },
        { id: 'e24', timestamp: 6000, agent: 'cto', type: 'tool-result', toolName: 'assess_technical_feasibility', result: 'Engineering: 2 person-months. Infrastructure: billing update. Feasibility: FEASIBLE. Debt: Manageable.' },
        { id: 'e25', timestamp: 6200, agent: 'hr-director', type: 'tool-result', toolName: 'assess_workforce_impact', result: 'Headcount: +3. Morale Risk: HIGH (CS/Sales burnout from churn). Retention Risk: ELEVATED.' },
        { id: 'e26', timestamp: 6500, agent: 'legal-counsel', type: 'tool-result', toolName: 'check_regulatory_compliance', result: 'Contract: MSA renewal caps at risk. Legal Review: 4 weeks. Litigation Risk: MODERATE.' },
        { id: 'e27', timestamp: 6800, agent: 'customer-advocate', type: 'tool-result', toolName: 'predict_customer_impact', result: 'NPS: 42 → 22 (-20). Churn Risk: 15%. Revenue at Risk: $7.5M. Proactive outreach critical.' },
        { id: 'e28', timestamp: 7000, agent: 'operations-lead', type: 'tool-result', toolName: 'analyze_supply_chain_impact', result: 'Lead Time: 4 weeks. Capacity: +10%. Process: Incremental. SLA Risk: MANAGEABLE.' },
        { id: 'e29', timestamp: 7200, agent: 'competitor-analyst', type: 'tool-result', toolName: 'predict_competitor_response', result: 'Competitors: Looker, Tableau. Response: undercut pricing. Market Share: -5%. First-Mover: NO.' },
        { id: 'e30', timestamp: 7500, agent: 'board-member', type: 'tool-result', toolName: 'evaluate_strategic_alignment', result: 'Vision: 5/10. Investor: NEUTRAL. Board Approval: Requires discussion. Governance: Routine.' },

        // Delegation results with stances
        { id: 'e31', timestamp: 8000, agent: 'cfo', type: 'delegation-result', source: 'coordinator', target: 'cfo', resultPreview: 'SUPPORT. Strong financial upside: +$10M ARR, 3-month payback, $22.5M NPV. Revenue growth justifies short-term churn risk.' },
        { id: 'e32', timestamp: 8300, agent: 'cto', type: 'delegation-result', source: 'coordinator', target: 'cto', resultPreview: 'SUPPORT. Technically feasible — only 2 person-months. Billing changes straightforward. Revenue funds AI infrastructure.' },
        { id: 'e33', timestamp: 8600, agent: 'cmo', type: 'delegation-result', source: 'coordinator', target: 'cmo', resultPreview: 'OPPOSE. Brand alignment only 4/10. High perception risk. Competitors will use this in marketing against us.' },
        { id: 'e34', timestamp: 8900, agent: 'hr-director', type: 'delegation-result', source: 'coordinator', target: 'hr-director', resultPreview: 'OPPOSE. CS/Sales morale at HIGH risk. Churn-driven burnout, missed quotas, elevated turnover. Need quota relief.' },
        { id: 'e35', timestamp: 9200, agent: 'legal-counsel', type: 'delegation-result', source: 'coordinator', target: 'legal-counsel', resultPreview: 'OPPOSE. Enterprise MSAs have renewal caps (5-10%). Blanket 25% likely breaches contracts. Full audit required.' },
        { id: 'e36', timestamp: 9500, agent: 'customer-advocate', type: 'delegation-result', source: 'coordinator', target: 'customer-advocate', resultPreview: 'OPPOSE. NPS drops from 42 to 22. 15% churn = $7.5M at risk. Enterprise customers will demand renegotiation.' },
        { id: 'e37', timestamp: 9800, agent: 'operations-lead', type: 'delegation-result', source: 'coordinator', target: 'operations-lead', resultPreview: 'SUPPORT. Operationally manageable — 4-week lead time. Billing update straightforward. Capacity impact minimal.' },
        { id: 'e38', timestamp: 10100, agent: 'competitor-analyst', type: 'delegation-result', source: 'coordinator', target: 'competitor-analyst', resultPreview: 'OPPOSE. Competitors will undercut. -5% market share expected. No first-mover advantage. Timing is wrong.' },
        { id: 'e39', timestamp: 10400, agent: 'board-member', type: 'delegation-result', source: 'coordinator', target: 'board-member', resultPreview: 'OPPOSE. Vision alignment only 5/10. Board requires modified approach. Prefer hybrid pricing strategy.' },

        // Coordinator delegates to analyst with all results
        { id: 'e40', timestamp: 11200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'analyst', channel: 'local', taskPreview: 'Synthesize 9 perspectives: 3 SUPPORT (CFO, CTO, Ops), 6 OPPOSE (CMO, HR, Legal, Customer, Competitor, Board). Generate recommendation.' },

        // Analyst synthesizes
        { id: 'e41', timestamp: 12000, agent: 'analyst', type: 'tool-call', toolName: 'synthesize_perspectives', args: 'total=9, support=3, oppose=6, neutral=0' },
        { id: 'e42', timestamp: 13000, agent: 'analyst', type: 'tool-result', toolName: 'synthesize_perspectives', result: 'Consensus: OPPOSITION MAJORITY. 3/6/0. Confidence: 87%. Key risks: churn ($7.5M), contract breaches, competitive response. Key opportunities: AI monetization via premium tier, inflation adjustment.' },
        { id: 'e43', timestamp: 14000, agent: 'analyst', type: 'tool-call', toolName: 'generate_recommendation', args: 'recommendation="proceed-with-conditions", conditions="Hybrid pricing: 7-10% base + AI premium tier"' },
        { id: 'e44', timestamp: 15000, agent: 'analyst', type: 'tool-result', toolName: 'generate_recommendation', result: 'RECOMMENDATION: PROCEED WITH CONDITIONS. Adopt hybrid model — 7-10% base increase + new AI premium tier. Audit MSAs. Protect frontline teams.' },
        { id: 'e45', timestamp: 15800, agent: 'analyst', type: 'delegation-result', source: 'coordinator', target: 'analyst', resultPreview: 'PROCEED WITH CONDITIONS. Reject blanket 25%, adopt hybrid pricing. 7-10% base + AI premium tier. Audit contracts first.' },

        // Final coordinator response
        { id: 'e46', timestamp: 17000, agent: 'coordinator', type: 'model-response', contentPreview: 'SIMULATION COMPLETE. 9 perspectives analyzed (3 Support, 6 Oppose). Recommendation: PROCEED WITH CONDITIONS — replace blanket 25% with hybrid strategy: 7-10% inflation adjustment + AI premium tier. Key risks: $7.5M churn, MSA breaches, competitive response. Immediate actions: audit contracts, design AI tier, update net-new pricing.' },
      ],
      totalDurationMs: 18000,
    },
    {
      id: 2,
      title: 'European Market Expansion',
      inputMessage:
        'We are evaluating expanding into the European market by opening a new data center in Frankfurt and hiring a 15-person EU sales team. Estimated investment: $8M over 18 months. Three competitors already operate in EU.',
      agents: [
        { id: 'coordinator', name: 'Coordinator', role: 'Simulation coordinator', tools: ['structure_decision', 'discover_agents', 'delegate_to'], color: 'blue', x: 8, y: 50 },
        { id: 'cfo', name: 'CFO', role: 'Financial analysis', tools: ['run_financial_model'], color: 'emerald', x: 32, y: 12 },
        { id: 'cmo', name: 'CMO', role: 'Marketing & brand', tools: ['analyze_brand_impact'], color: 'cyan', x: 50, y: 12 },
        { id: 'cto', name: 'CTO', role: 'Technical feasibility', tools: ['assess_technical_feasibility'], color: 'violet', x: 68, y: 12 },
        { id: 'hr-director', name: 'HR Director', role: 'Workforce impact', tools: ['assess_workforce_impact'], color: 'amber', x: 32, y: 50 },
        { id: 'legal-counsel', name: 'Legal', role: 'Regulatory compliance', tools: ['check_regulatory_compliance'], color: 'rose', x: 50, y: 50 },
        { id: 'customer-advocate', name: 'Customer', role: 'Customer impact', tools: ['predict_customer_impact'], color: 'yellow', x: 68, y: 50 },
        { id: 'operations-lead', name: 'Operations', role: 'Supply chain & ops', tools: ['analyze_supply_chain_impact'], color: 'emerald', x: 32, y: 88 },
        { id: 'competitor-analyst', name: 'Competitor', role: 'Competitive response', tools: ['predict_competitor_response'], color: 'cyan', x: 50, y: 88 },
        { id: 'board-member', name: 'Board', role: 'Strategic governance', tools: ['evaluate_strategic_alignment'], color: 'amber', x: 68, y: 88 },
        { id: 'analyst', name: 'Analyst', role: 'Synthesis & recommendation', tools: ['synthesize_perspectives', 'generate_recommendation'], color: 'rose', x: 92, y: 50 },
      ],
      events: [
        { id: 'f01', timestamp: 0, agent: 'coordinator', type: 'tool-call', toolName: 'structure_decision', args: 'decision="EU expansion — Frankfurt DC + 15-person sales team", time_horizon="18 months"' },
        { id: 'f02', timestamp: 800, agent: 'coordinator', type: 'tool-result', toolName: 'structure_decision', result: 'SIMULATION BRIEF: EU market expansion. $8M investment. SIM-20260324-7103. Status: READY FOR DISTRIBUTION.' },

        { id: 'f03', timestamp: 1500, agent: 'coordinator', type: 'discover-agents', capability: 'simulation', results: ['cfo', 'cmo', 'cto', 'hr-director', 'legal-counsel', 'customer-advocate', 'operations-lead', 'competitor-analyst', 'board-member'] },

        // Fan-out to all 9 personas
        { id: 'f04', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cfo', channel: 'local', taskPreview: 'Analyze EU expansion ($8M, 18mo) from financial perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f05', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cmo', channel: 'local', taskPreview: 'Analyze EU expansion from marketing/brand perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f06', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'cto', channel: 'local', taskPreview: 'Analyze EU expansion (Frankfurt DC) from technical perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f07', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'hr-director', channel: 'local', taskPreview: 'Analyze EU expansion (15 hires) from workforce perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f08', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'legal-counsel', channel: 'local', taskPreview: 'Analyze EU expansion from legal/GDPR compliance perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f09', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'customer-advocate', channel: 'local', taskPreview: 'Analyze EU expansion from customer experience perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f10', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'operations-lead', channel: 'local', taskPreview: 'Analyze EU expansion (new DC) from operations perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f11', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'competitor-analyst', channel: 'local', taskPreview: 'Analyze EU expansion from competitive landscape perspective. State SUPPORT/OPPOSE/NEUTRAL.' },
        { id: 'f12', timestamp: 2200, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'board-member', channel: 'local', taskPreview: 'Analyze EU expansion from strategic governance perspective. State SUPPORT/OPPOSE/NEUTRAL.' },

        // Persona tool calls
        { id: 'f13', timestamp: 3000, agent: 'cfo', type: 'tool-call', toolName: 'run_financial_model', args: 'revenue_impact_pct=+30, cost_impact_pct=+16, payback_months=18' },
        { id: 'f14', timestamp: 3300, agent: 'cmo', type: 'tool-call', toolName: 'analyze_brand_impact', args: 'brand_alignment_score=8, market_perception_risk="low"' },
        { id: 'f15', timestamp: 3600, agent: 'cto', type: 'tool-call', toolName: 'assess_technical_feasibility', args: 'engineering_months=14, infrastructure_changes="Frankfurt DC, GDPR infra"' },
        { id: 'f16', timestamp: 3900, agent: 'hr-director', type: 'tool-call', toolName: 'assess_workforce_impact', args: 'headcount_change=+15, reskilling_needed=false, morale_risk="low"' },
        { id: 'f17', timestamp: 4100, agent: 'legal-counsel', type: 'tool-call', toolName: 'check_regulatory_compliance', args: 'jurisdictions="EU/GDPR", data_privacy_impact=true' },
        { id: 'f18', timestamp: 4300, agent: 'customer-advocate', type: 'tool-call', toolName: 'predict_customer_impact', args: 'affected_segment="EU prospects", satisfaction_change=+15, churn_risk_pct=0' },
        { id: 'f19', timestamp: 4500, agent: 'operations-lead', type: 'tool-call', toolName: 'analyze_supply_chain_impact', args: 'vendor_changes="EU cloud provider", lead_time_weeks=12' },
        { id: 'f20', timestamp: 4700, agent: 'competitor-analyst', type: 'tool-call', toolName: 'predict_competitor_response', args: 'primary_competitors="3 EU incumbents", market_share_impact_pct=+8' },
        { id: 'f21', timestamp: 4900, agent: 'board-member', type: 'tool-call', toolName: 'evaluate_strategic_alignment', args: 'vision_alignment_score=9, investor_sentiment="positive"' },

        // Tool results
        { id: 'f22', timestamp: 5500, agent: 'cfo', type: 'tool-result', toolName: 'run_financial_model', result: 'Projected: +$15M ARR (30%). Cost: +$8M. Payback: 18mo. NPV (3yr): $29.5M. Risk: MODERATE.' },
        { id: 'f23', timestamp: 5700, agent: 'cmo', type: 'tool-result', toolName: 'analyze_brand_impact', result: 'Brand Alignment: 8/10. Global brand credibility. Low perception risk. Proactive announcement recommended.' },
        { id: 'f24', timestamp: 6000, agent: 'cto', type: 'tool-result', toolName: 'assess_technical_feasibility', result: 'Engineering: 14 person-months. Frankfurt DC + GDPR compliance. Feasibility: FEASIBLE. Debt: Manageable.' },
        { id: 'f25', timestamp: 6200, agent: 'hr-director', type: 'tool-result', toolName: 'assess_workforce_impact', result: 'Hiring: +15 (EU-based). Timeline: 30 weeks. Morale: LOW risk. Culture: growth signal. Retention: NORMAL.' },
        { id: 'f26', timestamp: 6400, agent: 'legal-counsel', type: 'tool-result', toolName: 'check_regulatory_compliance', result: 'GDPR: Full review required. DPO appointment needed. Data processing agreements. Timeline: 5 weeks.' },
        { id: 'f27', timestamp: 6600, agent: 'customer-advocate', type: 'tool-result', toolName: 'predict_customer_impact', result: 'NPS: +15 (EU data residency unlocks 200 leads/quarter). Zero churn risk. Net positive customer impact.' },
        { id: 'f28', timestamp: 6800, agent: 'operations-lead', type: 'tool-result', toolName: 'analyze_supply_chain_impact', result: 'New EU cloud vendor. Lead Time: 12 weeks. Capacity: +15%. SLA: MANAGEABLE. Rollback: LOW.' },
        { id: 'f29', timestamp: 7000, agent: 'competitor-analyst', type: 'tool-result', toolName: 'predict_competitor_response', result: 'EU incumbents will increase retention efforts. +8% market share expected. Strong first-mover in data residency.' },
        { id: 'f30', timestamp: 7200, agent: 'board-member', type: 'tool-result', toolName: 'evaluate_strategic_alignment', result: 'Vision: 9/10. Investor: POSITIVE. Board Approval: Likely. Geographic diversification is strategic priority.' },

        // Delegation results
        { id: 'f31', timestamp: 7800, agent: 'cfo', type: 'delegation-result', source: 'coordinator', target: 'cfo', resultPreview: 'SUPPORT. $29.5M NPV over 3 years. 200 EU leads/quarter is untapped revenue. 18-month payback acceptable.' },
        { id: 'f32', timestamp: 8100, agent: 'cmo', type: 'delegation-result', source: 'coordinator', target: 'cmo', resultPreview: 'SUPPORT. Strong brand alignment (8/10). EU presence signals global credibility. Low-risk messaging.' },
        { id: 'f33', timestamp: 8400, agent: 'cto', type: 'delegation-result', source: 'coordinator', target: 'cto', resultPreview: 'SUPPORT. Technically feasible. 14 person-months, 5 engineers. Frankfurt DC is well-supported region.' },
        { id: 'f34', timestamp: 8700, agent: 'hr-director', type: 'delegation-result', source: 'coordinator', target: 'hr-director', resultPreview: 'SUPPORT. Growth hiring boosts morale. EU talent pool is strong. Low retention risk.' },
        { id: 'f35', timestamp: 9000, agent: 'legal-counsel', type: 'delegation-result', source: 'coordinator', target: 'legal-counsel', resultPreview: 'NEUTRAL. GDPR compliance is achievable but requires dedicated DPO, 5-week legal review, and ongoing audit.' },
        { id: 'f36', timestamp: 9300, agent: 'customer-advocate', type: 'delegation-result', source: 'coordinator', target: 'customer-advocate', resultPreview: 'SUPPORT. EU data residency unlocks 200 leads/quarter. Net positive NPS. Zero churn risk.' },
        { id: 'f37', timestamp: 9600, agent: 'operations-lead', type: 'delegation-result', source: 'coordinator', target: 'operations-lead', resultPreview: 'NEUTRAL. Operationally manageable but 12-week lead time for DC setup. Need EU cloud vendor evaluation.' },
        { id: 'f38', timestamp: 9900, agent: 'competitor-analyst', type: 'delegation-result', source: 'coordinator', target: 'competitor-analyst', resultPreview: 'SUPPORT. First-mover on data residency positioning. +8% EU market share. Competitors will scramble to retain.' },
        { id: 'f39', timestamp: 10200, agent: 'board-member', type: 'delegation-result', source: 'coordinator', target: 'board-member', resultPreview: 'SUPPORT. Vision alignment 9/10. Geographic diversification is board priority. Investors will react positively.' },

        // Coordinator delegates to analyst
        { id: 'f40', timestamp: 11000, agent: 'coordinator', type: 'delegation-request', source: 'coordinator', target: 'analyst', channel: 'local', taskPreview: 'Synthesize 9 perspectives: 7 SUPPORT, 2 NEUTRAL, 0 OPPOSE. Generate recommendation for EU expansion.' },

        { id: 'f41', timestamp: 11800, agent: 'analyst', type: 'tool-call', toolName: 'synthesize_perspectives', args: 'total=9, support=7, oppose=0, neutral=2' },
        { id: 'f42', timestamp: 12800, agent: 'analyst', type: 'tool-result', toolName: 'synthesize_perspectives', result: 'Consensus: STRONG SUPPORT. 7/0/2. Confidence: 87%. Key opportunity: $29.5M NPV, 200 leads/quarter. Key risk: GDPR compliance timeline, DC lead time.' },
        { id: 'f43', timestamp: 13500, agent: 'analyst', type: 'tool-call', toolName: 'generate_recommendation', args: 'recommendation="proceed", conditions="GDPR readiness before launch, DPO appointment"' },
        { id: 'f44', timestamp: 14500, agent: 'analyst', type: 'tool-result', toolName: 'generate_recommendation', result: 'RECOMMENDATION: PROCEED. Strong consensus (7/9 support). Begin Frankfurt DC procurement and EU hiring immediately. GDPR readiness is gate for customer onboarding.' },
        { id: 'f45', timestamp: 15200, agent: 'analyst', type: 'delegation-result', source: 'coordinator', target: 'analyst', resultPreview: 'PROCEED. Strong consensus (7 Support, 2 Neutral, 0 Oppose). Begin immediately. GDPR compliance is the only gate.' },

        { id: 'f46', timestamp: 16500, agent: 'coordinator', type: 'model-response', contentPreview: 'SIMULATION COMPLETE. 9 perspectives analyzed (7 Support, 2 Neutral, 0 Oppose). Recommendation: PROCEED. Strong strategic alignment, $29.5M NPV, 200 EU leads unlocked. Conditions: appoint DPO, complete GDPR review before customer onboarding. Immediate actions: begin DC procurement, start EU hiring pipeline.' },
      ],
      totalDurationMs: 17500,
    },
  ],
}
