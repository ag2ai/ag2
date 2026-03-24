import type { ReplayData } from '../replay-types'

export const replay09: ReplayData = {
  example: {
    id: '09',
    title: 'GTM Sales',
    category: 'Pipeline Orchestration',
    description: 'Five agents coordinate a sales cycle with PipelineGuard enforcing valid routing.',
    themes: ['network', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Enterprise Lead Qualification',
      inputMessage: 'New inbound lead: CTO of MegaCorp (5,000 employees) requested a demo after reading our whitepaper on AI automation.',
      agents: [
        { id: 'sdr', name: 'SDR', role: 'Sales Development', tools: ['enrich_lead', 'score_lead', 'schedule_call'], color: 'cyan', x: 10, y: 50 },
        { id: 'ae', name: 'AE', role: 'Account Executive', tools: ['research_account', 'create_proposal', 'schedule_demo'], color: 'blue', x: 30, y: 50 },
        { id: 'se', name: 'SE', role: 'Solutions Engineer', tools: ['assess_technical_fit', 'build_poc', 'create_architecture'], color: 'violet', x: 50, y: 50 },
        { id: 'marketing', name: 'Marketing', role: 'Content & campaigns', tools: ['get_engagement_data', 'create_nurture_sequence'], color: 'rose', x: 70, y: 50 },
        { id: 'cs', name: 'CS', role: 'Customer Success', tools: ['create_onboarding_plan', 'assign_csm'], color: 'emerald', x: 90, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'sdr', type: 'tool-call', toolName: 'enrich_lead', args: 'company="MegaCorp", title="CTO"' },
        { id: 'e02', timestamp: 1200, agent: 'sdr', type: 'tool-result', toolName: 'enrich_lead', result: 'MegaCorp: 5,000 emp, $2.1B revenue, Series D. CTO: James Park, ex-Google. Tech stack: AWS, Python, Kubernetes.' },
        { id: 'e03', timestamp: 2200, agent: 'sdr', type: 'tool-call', toolName: 'score_lead', args: 'company_size=5000, title="CTO", source="whitepaper"' },
        { id: 'e04', timestamp: 3000, agent: 'sdr', type: 'tool-result', toolName: 'score_lead', result: 'Lead score: 92/100 (HOT). Enterprise tier. Whitepaper download = high intent. CTO = decision maker.' },
        { id: 'e05', timestamp: 4000, agent: 'sdr', type: 'delegation-request', source: 'sdr', target: 'ae', channel: 'local', taskPreview: 'HOT lead: James Park, CTO of MegaCorp (5K emp, $2.1B). Score: 92. Requested demo after whitepaper. Enterprise tier.' },
        { id: 'e06', timestamp: 5500, agent: 'ae', type: 'tool-call', toolName: 'research_account', args: 'company="MegaCorp"' },
        { id: 'e07', timestamp: 6500, agent: 'ae', type: 'tool-result', toolName: 'research_account', result: 'MegaCorp: AI automation interest. Current pain: manual workflow overhead. Budget cycle: Q2. Competitor: using basic RPA tools.' },
        { id: 'e08', timestamp: 7500, agent: 'ae', type: 'delegation-request', source: 'ae', target: 'se', channel: 'local', taskPreview: 'Need technical assessment for MegaCorp. AWS/Python/K8s stack. Pain: manual workflows. Assess fit for our AI automation platform.' },
        { id: 'e09', timestamp: 9000, agent: 'se', type: 'tool-call', toolName: 'assess_technical_fit', args: 'stack="AWS, Python, Kubernetes", use_case="workflow automation"' },
        { id: 'e10', timestamp: 10000, agent: 'se', type: 'tool-result', toolName: 'assess_technical_fit', result: 'Technical fit: EXCELLENT (95%). K8s deployment supported. Python SDK available. AWS integration native. Estimated 10x improvement over current RPA.' },
        { id: 'e11', timestamp: 11000, agent: 'se', type: 'delegation-result', source: 'ae', target: 'se', resultPreview: 'Technical fit: EXCELLENT. Full stack compatibility. Ready for POC.' },
        { id: 'e12', timestamp: 12000, agent: 'ae', type: 'tool-call', toolName: 'schedule_demo', args: 'contact="James Park", company="MegaCorp", type="enterprise"' },
        { id: 'e13', timestamp: 13000, agent: 'ae', type: 'tool-result', toolName: 'schedule_demo', result: 'Demo scheduled: Thursday 2pm PT. Attendees: CTO, VP Eng, 2 senior engineers. Custom agenda: AI automation for K8s workflows.' },
        { id: 'e14', timestamp: 13500, agent: 'ae', type: 'delegation-result', source: 'sdr', target: 'ae', resultPreview: 'Enterprise demo scheduled Thursday 2pm. Technical fit excellent. MegaCorp moving to next stage.' },
        { id: 'e15', timestamp: 14500, agent: 'sdr', type: 'model-response', contentPreview: 'Lead qualified and advanced. MegaCorp (CTO James Park): Score 92, technical fit 95%. Enterprise demo scheduled Thursday 2pm with CTO, VP Eng, and 2 senior engineers. Custom agenda on AI automation for their K8s workflows. Pipeline value: ~$500K ARR.' },
      ],
      totalDurationMs: 15500,
    },
  ],
}
