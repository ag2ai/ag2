import type { ReplayData } from '../replay-types'

export const replay03: ReplayData = {
  example: {
    id: '03',
    title: 'Newsroom Pipeline',
    category: 'Multi-Agent via Hub',
    description: 'Four agents collaborate through a Hub to research, write, edit, and publish an article.',
    themes: ['network'],
  },
  scenarios: [
    {
      id: 1,
      title: 'AI Agent Frameworks',
      inputMessage: 'Write an article about the rise of AI agent frameworks and how they\'re changing software development',
      agents: [
        { id: 'researcher', name: 'Researcher', role: 'Senior Research Journalist', tools: ['search_web', 'analyze_sources'], color: 'cyan', x: 15, y: 50 },
        { id: 'writer', name: 'Writer', role: 'Staff Writer', tools: ['draft_article', 'check_word_count'], color: 'yellow', x: 38, y: 50 },
        { id: 'editor', name: 'Editor', role: 'Senior Editor', tools: ['review_article', 'check_grammar', 'verify_facts'], color: 'rose', x: 62, y: 50 },
        { id: 'publisher', name: 'Publisher', role: 'Publishing Manager', tools: ['format_for_web', 'generate_social_posts', 'schedule_publication'], color: 'emerald', x: 85, y: 50 },
      ],
      events: [
        // Researcher phase
        { id: 'e01', timestamp: 0, agent: 'researcher', type: 'tool-call', toolName: 'search_web', args: 'query="AI agent frameworks 2025"' },
        { id: 'e02', timestamp: 1500, agent: 'researcher', type: 'tool-result', toolName: 'search_web', result: '[1] The Rise of AI Agent Frameworks in 2025 — market grew 340% [2] Enterprise Adoption — 67% of Fortune 500 [3] Developer Survey — 73% plan to integrate agents' },
        { id: 'e03', timestamp: 3000, agent: 'researcher', type: 'tool-call', toolName: 'analyze_sources', args: 'topic="AI agent frameworks"' },
        { id: 'e04', timestamp: 4200, agent: 'researcher', type: 'tool-result', toolName: 'analyze_sources', result: 'Key findings: rapid growth 300%+ YoY, enterprise adoption at 67%, developer interest at all-time high. Strong basis for feature article.' },
        { id: 'e05', timestamp: 5500, agent: 'researcher', type: 'discover-agents', capability: 'writing', results: ['writer'] },
        { id: 'e06', timestamp: 6500, agent: 'researcher', type: 'delegation-request', source: 'researcher', target: 'writer', channel: 'local', taskPreview: 'Write an article about AI agent frameworks. Key findings: 340% market growth, 67% Fortune 500 adoption, top concerns: reliability, cost, observability...' },

        // Writer phase
        { id: 'e07', timestamp: 8000, agent: 'writer', type: 'tool-call', toolName: 'draft_article', args: 'headline="The Rise of AI Agent Frameworks", tone="informative"' },
        { id: 'e08', timestamp: 9500, agent: 'writer', type: 'tool-result', toolName: 'draft_article', result: '# The Rise of AI Agent Frameworks — The landscape is shifting beneath our feet. What was once the domain of research labs...' },
        { id: 'e09', timestamp: 10500, agent: 'writer', type: 'tool-call', toolName: 'check_word_count', args: 'text="[article draft]"' },
        { id: 'e10', timestamp: 11000, agent: 'writer', type: 'tool-result', toolName: 'check_word_count', result: 'Word count: 287 words. Good length.' },
        { id: 'e11', timestamp: 12000, agent: 'writer', type: 'discover-agents', capability: 'editing', results: ['editor'] },
        { id: 'e12', timestamp: 13000, agent: 'writer', type: 'delegation-request', source: 'writer', target: 'editor', channel: 'local', taskPreview: 'Please review this article: "The Rise of AI Agent Frameworks" — 287 words, informative tone, covers market growth and enterprise adoption...' },

        // Editor phase
        { id: 'e13', timestamp: 14500, agent: 'editor', type: 'tool-call', toolName: 'review_article', args: 'article_text="[full draft]"' },
        { id: 'e14', timestamp: 15800, agent: 'editor', type: 'tool-result', toolName: 'review_article', result: 'STRONG — publishable with minor revisions. Good data, balanced perspective, clear narrative arc.' },
        { id: 'e15', timestamp: 16800, agent: 'editor', type: 'tool-call', toolName: 'check_grammar', args: 'text="[full draft]"' },
        { id: 'e16', timestamp: 17500, agent: 'editor', type: 'tool-result', toolName: 'check_grammar', result: 'Issues: 2 minor. Readability: 72/100. Spelling: clean. Tone consistency: PASS' },
        { id: 'e17', timestamp: 18500, agent: 'editor', type: 'tool-call', toolName: 'verify_facts', args: 'claims="300% growth, 67% Fortune 500, historic developer interest"' },
        { id: 'e18', timestamp: 19500, agent: 'editor', type: 'tool-result', toolName: 'verify_facts', result: 'All claims VERIFIED. 300% growth confirmed, 67% adoption confirmed, developer interest confirmed by Stack Overflow data.' },
        { id: 'e19', timestamp: 20500, agent: 'editor', type: 'discover-agents', capability: 'publishing', results: ['publisher'] },
        { id: 'e20', timestamp: 21500, agent: 'editor', type: 'delegation-request', source: 'editor', target: 'publisher', channel: 'local', taskPreview: 'Article approved for publication: "The Rise of AI Agent Frameworks". All facts verified, grammar checked, minor edits applied.' },

        // Publisher phase
        { id: 'e21', timestamp: 23000, agent: 'publisher', type: 'tool-call', toolName: 'format_for_web', args: 'headline="The Rise of AI Agent Frameworks"' },
        { id: 'e22', timestamp: 24000, agent: 'publisher', type: 'tool-result', toolName: 'format_for_web', result: '<article class="feature-story">...</article> — SEO tags, Open Graph metadata, responsive layout applied.' },
        { id: 'e23', timestamp: 25000, agent: 'publisher', type: 'tool-call', toolName: 'generate_social_posts', args: 'headline="The Rise of AI Agent Frameworks"' },
        { id: 'e24', timestamp: 26000, agent: 'publisher', type: 'tool-result', toolName: 'generate_social_posts', result: 'Twitter/X, LinkedIn, and Bluesky posts generated.' },
        { id: 'e25', timestamp: 27000, agent: 'publisher', type: 'tool-call', toolName: 'schedule_publication', args: 'publish_time="immediate"' },
        { id: 'e26', timestamp: 28000, agent: 'publisher', type: 'tool-result', toolName: 'schedule_publication', result: 'PUB-20260324-001 | Status: LIVE | Distribution: Website, RSS, newsletter | Social: queued' },

        // Results flow back
        { id: 'e27', timestamp: 29000, agent: 'publisher', type: 'delegation-result', source: 'editor', target: 'publisher', resultPreview: 'Article published. PUB-20260324-001 is LIVE.' },
        { id: 'e28', timestamp: 29500, agent: 'editor', type: 'delegation-result', source: 'writer', target: 'editor', resultPreview: 'Article published and distributed across all channels.' },
        { id: 'e29', timestamp: 30000, agent: 'writer', type: 'delegation-result', source: 'researcher', target: 'writer', resultPreview: 'Pipeline complete. Article live on website with social promotion.' },
        { id: 'e30', timestamp: 31000, agent: 'researcher', type: 'model-response', contentPreview: 'Pipeline complete! Article "The Rise of AI Agent Frameworks" is now LIVE (PUB-20260324-001). Published to website, RSS feed, and newsletter. Social media posts queued for Twitter/X, LinkedIn, and Bluesky.' },
      ],
      totalDurationMs: 32000,
    },
  ],
}
