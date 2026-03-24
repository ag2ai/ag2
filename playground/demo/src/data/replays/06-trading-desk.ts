import type { ReplayData } from '../replay-types'

export const replay06: ReplayData = {
  example: {
    id: '06',
    title: 'Trading Desk',
    category: 'Full Customization',
    description: 'Custom priority schemes, events, routing plugins (RiskGate, RateLimiter), and system plugins.',
    themes: ['network', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Market Analysis',
      inputMessage: 'Analyze current market conditions for NVDA and recommend position sizing.',
      agents: [
        { id: 'analyst', name: 'Analyst', role: 'Market analyst', tools: ['get_market_data', 'run_technical_analysis', 'get_news_sentiment'], color: 'cyan', x: 20, y: 30 },
        { id: 'trader', name: 'Trader', role: 'Trade execution', tools: ['get_position', 'execute_trade', 'check_limits'], color: 'blue', x: 50, y: 30 },
        { id: 'risk-mgr', name: 'Risk Manager', role: 'Risk assessment', tools: ['calculate_var', 'check_exposure', 'stress_test'], color: 'rose', x: 50, y: 70 },
        { id: 'compliance', name: 'Compliance', role: 'Regulatory compliance', tools: ['check_restrictions', 'log_audit_trail'], color: 'amber', x: 80, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'analyst', type: 'tool-call', toolName: 'get_market_data', args: 'symbol="NVDA"' },
        { id: 'e02', timestamp: 1200, agent: 'analyst', type: 'tool-result', toolName: 'get_market_data', result: 'NVDA: $892.45 (+2.3%) | Vol: 45M | RSI: 68 | 50MA: $845 | 200MA: $720 | Sector: +1.8%' },
        { id: 'e03', timestamp: 2200, agent: 'analyst', type: 'tool-call', toolName: 'run_technical_analysis', args: 'symbol="NVDA", timeframe="1D"' },
        { id: 'e04', timestamp: 3500, agent: 'analyst', type: 'tool-result', toolName: 'run_technical_analysis', result: 'Trend: BULLISH. Above both MAs. RSI approaching overbought. Support: $860. Resistance: $910. Signal: BUY with caution.' },
        { id: 'e05', timestamp: 4500, agent: 'analyst', type: 'tool-call', toolName: 'get_news_sentiment', args: 'symbol="NVDA"' },
        { id: 'e06', timestamp: 5500, agent: 'analyst', type: 'tool-result', toolName: 'get_news_sentiment', result: 'Sentiment: POSITIVE (0.78). Key: new GPU architecture announcement, strong data center demand, AI infrastructure spending.' },
        { id: 'e07', timestamp: 6500, agent: 'analyst', type: 'delegation-request', source: 'analyst', target: 'trader', channel: 'local', taskPreview: 'NVDA is BULLISH. Recommend buying. Check current position and execute if within risk limits.' },
        // RiskGate plugin processes the delegation
        { id: 'e08', timestamp: 7500, agent: 'trader', type: 'tool-call', toolName: 'check_limits', args: 'symbol="NVDA", action="BUY"' },
        { id: 'e09', timestamp: 8300, agent: 'trader', type: 'tool-result', toolName: 'check_limits', result: 'NVDA: current position 500 shares ($446K). Sector limit: $1M. Portfolio limit: $5M. Available: $554K. WITHIN LIMITS.' },
        { id: 'e10', timestamp: 9300, agent: 'trader', type: 'delegation-request', source: 'trader', target: 'risk-mgr', channel: 'local', taskPreview: 'Requesting risk assessment for adding 200 shares NVDA at ~$892.' },
        { id: 'e11', timestamp: 10300, agent: 'risk-mgr', type: 'tool-call', toolName: 'calculate_var', args: 'position="NVDA +200 shares"' },
        { id: 'e12', timestamp: 11300, agent: 'risk-mgr', type: 'tool-result', toolName: 'calculate_var', result: 'VaR (95%, 1-day): $12,400. Stress scenario: -$28,900. Within acceptable thresholds.' },
        { id: 'e13', timestamp: 12300, agent: 'risk-mgr', type: 'delegation-result', source: 'trader', target: 'risk-mgr', resultPreview: 'APPROVED. VaR within limits. Proceed with caution — RSI near overbought.' },
        { id: 'e14', timestamp: 13000, agent: 'trader', type: 'tool-call', toolName: 'execute_trade', args: 'symbol="NVDA", action="BUY", quantity=200' },
        { id: 'e15', timestamp: 14000, agent: 'trader', type: 'tool-result', toolName: 'execute_trade', result: 'EXECUTED: BUY 200 NVDA @ $892.45. Total cost: $178,490. New position: 700 shares ($624,715).' },
        { id: 'e16', timestamp: 14500, agent: 'compliance', type: 'tool-call', toolName: 'log_audit_trail', args: 'event="TRADE_EXECUTED", details="BUY 200 NVDA"' },
        { id: 'e17', timestamp: 15200, agent: 'compliance', type: 'tool-result', toolName: 'log_audit_trail', result: 'Audit entry logged: TRD-20260324-0042. Compliance check: PASS. No restricted list match.' },
        { id: 'e18', timestamp: 16000, agent: 'trader', type: 'delegation-result', source: 'analyst', target: 'trader', resultPreview: 'Trade executed: BUY 200 NVDA @ $892.45. Risk approved, compliance logged.' },
        { id: 'e19', timestamp: 17000, agent: 'analyst', type: 'model-response', contentPreview: 'Analysis complete for NVDA. Bullish trend confirmed — bought 200 shares @ $892.45 ($178,490). Position now 700 shares. Risk approved (VaR $12.4K), compliance logged. Watch RSI at 68 — set stop-loss at $860 support.' },
      ],
      totalDurationMs: 18000,
    },
  ],
}
