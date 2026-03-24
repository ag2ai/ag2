import type { ReplayData } from '../replay-types'

export const replay10: ReplayData = {
  example: {
    id: '10',
    title: 'Hiring Pipeline',
    category: 'Fanout + Conditional',
    description: 'Parallel screening via Fanout topology, then Conditional routing based on results.',
    themes: ['network'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Strong Senior Engineer',
      inputMessage: 'New candidate: Alex Rivera, applying for Senior Backend Engineer. 8 years experience, ex-Stripe, MIT CS degree.',
      agents: [
        { id: 'recruiter', name: 'Recruiter', role: 'Pipeline coordinator', tools: ['parse_resume', 'check_pipeline_status'], color: 'blue', x: 15, y: 50 },
        { id: 'tech-screen', name: 'Technical', role: 'Technical screener', tools: ['evaluate_skills', 'check_github'], color: 'violet', x: 50, y: 20 },
        { id: 'culture-screen', name: 'Culture', role: 'Culture fit screener', tools: ['evaluate_values', 'check_references'], color: 'cyan', x: 50, y: 50 },
        { id: 'bg-check', name: 'Background', role: 'Background checker', tools: ['verify_employment', 'verify_education'], color: 'amber', x: 50, y: 80 },
        { id: 'hiring-mgr', name: 'Hiring Manager', role: 'Final decision', tools: ['review_candidate', 'make_offer', 'schedule_onsite'], color: 'emerald', x: 85, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'recruiter', type: 'tool-call', toolName: 'parse_resume', args: 'candidate="Alex Rivera"' },
        { id: 'e02', timestamp: 1000, agent: 'recruiter', type: 'tool-result', toolName: 'parse_resume', result: 'Alex Rivera: 8yr exp, Stripe (4yr), DataDog (3yr), startup (1yr). MIT CS 2018. Skills: Go, Python, PostgreSQL, K8s, distributed systems.' },

        // Fanout: 3 parallel screenings
        { id: 'e03', timestamp: 2000, agent: 'recruiter', type: 'delegation-request', source: 'recruiter', target: 'tech-screen', channel: 'local', taskPreview: '[FANOUT 1/3] Technical screening for Alex Rivera. 8yr backend, ex-Stripe. Skills: Go, Python, distributed systems.' },
        { id: 'e04', timestamp: 2000, agent: 'recruiter', type: 'delegation-request', source: 'recruiter', target: 'culture-screen', channel: 'local', taskPreview: '[FANOUT 2/3] Culture fit screening for Alex Rivera. Ex-Stripe/DataDog. Evaluate collaboration and values alignment.' },
        { id: 'e05', timestamp: 2000, agent: 'recruiter', type: 'delegation-request', source: 'recruiter', target: 'bg-check', channel: 'local', taskPreview: '[FANOUT 3/3] Background verification for Alex Rivera. Employment: Stripe, DataDog. Education: MIT CS 2018.' },

        // Technical screening (parallel)
        { id: 'e06', timestamp: 3000, agent: 'tech-screen', type: 'tool-call', toolName: 'evaluate_skills', args: 'candidate="Alex Rivera", role="Senior Backend"' },
        { id: 'e07', timestamp: 4500, agent: 'tech-screen', type: 'tool-result', toolName: 'evaluate_skills', result: 'Technical score: 9.2/10. Strong distributed systems (Stripe-scale). Go expertise. System design: excellent. Algorithms: strong.' },
        { id: 'e08', timestamp: 5500, agent: 'tech-screen', type: 'tool-call', toolName: 'check_github', args: 'username="arivera-dev"' },
        { id: 'e09', timestamp: 6500, agent: 'tech-screen', type: 'tool-result', toolName: 'check_github', result: 'GitHub: 2.3K contributions/yr. Notable: maintainer of go-distributed-cache (1.2K stars). Clean code, good documentation.' },
        { id: 'e10', timestamp: 7500, agent: 'tech-screen', type: 'delegation-result', source: 'recruiter', target: 'tech-screen', resultPreview: 'STRONG PASS. Technical: 9.2/10. OSS maintainer. Stripe-scale distributed systems experience.' },

        // Culture screening (parallel)
        { id: 'e11', timestamp: 3500, agent: 'culture-screen', type: 'tool-call', toolName: 'evaluate_values', args: 'candidate="Alex Rivera"' },
        { id: 'e12', timestamp: 5000, agent: 'culture-screen', type: 'tool-result', toolName: 'evaluate_values', result: 'Values alignment: HIGH. Collaborative (team lead at Stripe), mentoring experience, open-source contributor. Communication: articulate.' },
        { id: 'e13', timestamp: 6000, agent: 'culture-screen', type: 'tool-call', toolName: 'check_references', args: 'candidate="Alex Rivera"' },
        { id: 'e14', timestamp: 7200, agent: 'culture-screen', type: 'tool-result', toolName: 'check_references', result: '2/2 references positive. "Best engineer I\'ve worked with" — former Stripe manager. "Great mentor" — junior engineer at DataDog.' },
        { id: 'e15', timestamp: 8000, agent: 'culture-screen', type: 'delegation-result', source: 'recruiter', target: 'culture-screen', resultPreview: 'STRONG PASS. High values alignment. Excellent references from Stripe and DataDog.' },

        // Background check (parallel)
        { id: 'e16', timestamp: 3000, agent: 'bg-check', type: 'tool-call', toolName: 'verify_employment', args: 'candidate="Alex Rivera", companies=["Stripe", "DataDog"]' },
        { id: 'e17', timestamp: 4800, agent: 'bg-check', type: 'tool-result', toolName: 'verify_employment', result: 'Stripe: VERIFIED (2020-2024, Senior Engineer → Staff Engineer). DataDog: VERIFIED (2017-2020, Engineer → Senior).' },
        { id: 'e18', timestamp: 5800, agent: 'bg-check', type: 'tool-call', toolName: 'verify_education', args: 'candidate="Alex Rivera", school="MIT"' },
        { id: 'e19', timestamp: 6800, agent: 'bg-check', type: 'tool-result', toolName: 'verify_education', result: 'MIT: VERIFIED. BS Computer Science, 2018. Honors: magna cum laude.' },
        { id: 'e20', timestamp: 7500, agent: 'bg-check', type: 'delegation-result', source: 'recruiter', target: 'bg-check', resultPreview: 'ALL VERIFIED. Employment and education confirmed. No discrepancies.' },

        // Conditional routing: Strong match → fast-track to hiring manager
        { id: 'e21', timestamp: 9000, agent: 'recruiter', type: 'delegation-request', source: 'recruiter', target: 'hiring-mgr', channel: 'local', taskPreview: '[FAST-TRACK] All 3 screenings STRONG PASS. Alex Rivera: Tech 9.2/10, culture HIGH, background VERIFIED. Routing to hiring manager.' },
        { id: 'e22', timestamp: 10000, agent: 'hiring-mgr', type: 'tool-call', toolName: 'review_candidate', args: 'candidate="Alex Rivera", screening="all-pass"' },
        { id: 'e23', timestamp: 11000, agent: 'hiring-mgr', type: 'tool-result', toolName: 'review_candidate', result: 'Candidate review: EXCEPTIONAL. Meets all requirements. Fast-track approved. Recommend: skip phone screen, proceed to onsite.' },
        { id: 'e24', timestamp: 12000, agent: 'hiring-mgr', type: 'tool-call', toolName: 'schedule_onsite', args: 'candidate="Alex Rivera", format="fast-track"' },
        { id: 'e25', timestamp: 13000, agent: 'hiring-mgr', type: 'tool-result', toolName: 'schedule_onsite', result: 'Onsite scheduled: Next Tuesday. 4 interviews: system design, coding, architecture review, team lunch. Panel: 3 senior engineers + VP Eng.' },
        { id: 'e26', timestamp: 13500, agent: 'hiring-mgr', type: 'delegation-result', source: 'recruiter', target: 'hiring-mgr', resultPreview: 'Fast-tracked. Onsite next Tuesday. Full interview panel arranged.' },
        { id: 'e27', timestamp: 14500, agent: 'recruiter', type: 'model-response', contentPreview: 'Alex Rivera: FAST-TRACKED. All 3 parallel screenings passed (Technical: 9.2/10, Culture: HIGH, Background: VERIFIED). Hiring manager approved — onsite scheduled next Tuesday with system design, coding, architecture, and team lunch. Panel: 3 senior engineers + VP Eng.' },
      ],
      totalDurationMs: 15500,
    },
  ],
}
