import type { ReplayData } from '../replay-types'

export const replay08: ReplayData = {
  example: {
    id: '08',
    title: 'Safety Harness',
    category: 'Observer Guardrails',
    description: 'Multiple observers monitor budget, detect loops, enforce content policies, and prevent tool abuse.',
    themes: ['autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Budget Enforcement',
      inputMessage: 'Research everything about quantum computing and write a comprehensive 10,000-word report.',
      agents: [
        { id: 'researcher', name: 'Researcher', role: 'Research agent with guardrails', tools: ['web_search', 'read_paper', 'write_section'], color: 'violet', x: 50, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'researcher', type: 'tool-call', toolName: 'web_search', args: 'query="quantum computing 2025 overview"' },
        { id: 'e02', timestamp: 1200, agent: 'researcher', type: 'tool-result', toolName: 'web_search', result: '10 results found covering quantum hardware, algorithms, error correction, and commercial applications.' },
        { id: 'e03', timestamp: 2500, agent: 'researcher', type: 'tool-call', toolName: 'web_search', args: 'query="quantum error correction breakthroughs"' },
        { id: 'e04', timestamp: 3500, agent: 'researcher', type: 'tool-result', toolName: 'web_search', result: '8 results on topological qubits, surface codes, and logical error rates.' },
        { id: 'e05', timestamp: 5000, agent: 'researcher', type: 'tool-call', toolName: 'write_section', args: 'section="Introduction to Quantum Computing"' },
        { id: 'e06', timestamp: 6500, agent: 'researcher', type: 'tool-result', toolName: 'write_section', result: 'Section written: 850 words covering fundamentals, qubit types, and current state.' },
        { id: 'e07', timestamp: 8000, agent: 'researcher', type: 'tool-call', toolName: 'web_search', args: 'query="quantum computing commercial applications"' },
        { id: 'e08', timestamp: 9000, agent: 'researcher', type: 'tool-result', toolName: 'web_search', result: '12 results on pharma, finance, materials science, cryptography applications.' },
        { id: 'e09', timestamp: 10500, agent: 'researcher', type: 'model-response', contentPreview: '[TokenMonitor WARNING] Token usage at 8,500/10,000 (85%). Wrapping up research. Generated introduction section (850 words). Key areas identified but budget insufficient for full 10,000-word report. Delivering focused 2,000-word summary instead.' },
      ],
      totalDurationMs: 11500,
    },
  ],
}
