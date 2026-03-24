import type { ReplayData } from '../replay-types'

export const replay02: ReplayData = {
  example: {
    id: '02',
    title: 'SRE Bot',
    category: 'Autonomous Monitoring',
    description: 'An autonomous agent with scheduled health checks and incident investigation.',
    themes: ['autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Periodic Health Checks',
      inputMessage: '(Autonomous — triggered by IntervalWatch every 10 seconds)',
      agents: [
        { id: 'sre', name: 'SRE Bot', role: 'Site Reliability Engineer', tools: ['check_service_health', 'restart_service', 'get_logs', 'list_alerts', 'create_incident'], color: 'emerald', x: 50, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'sre', type: 'scheduler-trigger', target: 'sre', watch: 'IntervalWatch(10s)', task: 'Run periodic health check on all services' },
        { id: 'e02', timestamp: 500, agent: 'sre', type: 'tool-call', toolName: 'check_service_health', args: 'service="all"' },
        { id: 'e03', timestamp: 1500, agent: 'sre', type: 'tool-result', toolName: 'check_service_health', result: 'api-gateway: HEALTHY | auth-service: HEALTHY | db-primary: DEGRADED (latency 340ms) | cache-redis: HEALTHY | worker-pool: HEALTHY' },
        { id: 'e04', timestamp: 2500, agent: 'sre', type: 'tool-call', toolName: 'get_logs', args: 'service="db-primary", lines=20' },
        { id: 'e05', timestamp: 3500, agent: 'sre', type: 'tool-result', toolName: 'get_logs', result: '[WARN] Slow query detected: SELECT * FROM orders WHERE ... (287ms). Connection pool at 89% capacity. Checkpoint in progress.' },
        { id: 'e06', timestamp: 4500, agent: 'sre', type: 'tool-call', toolName: 'list_alerts', args: 'severity="warning"' },
        { id: 'e07', timestamp: 5200, agent: 'sre', type: 'tool-result', toolName: 'list_alerts', result: 'Active alerts: 1 — "db-primary latency > 200ms" triggered 4 min ago.' },
        { id: 'e08', timestamp: 6500, agent: 'sre', type: 'model-response', contentPreview: 'Health check complete. 4/5 services healthy. db-primary is DEGRADED — slow queries detected, connection pool at 89%. Likely cause: checkpoint operation. Monitoring — will escalate if latency persists beyond next check cycle.' },

        // Second trigger cycle
        { id: 'e09', timestamp: 10000, agent: 'sre', type: 'scheduler-trigger', target: 'sre', watch: 'IntervalWatch(10s)', task: 'Run periodic health check on all services' },
        { id: 'e10', timestamp: 10500, agent: 'sre', type: 'tool-call', toolName: 'check_service_health', args: 'service="all"' },
        { id: 'e11', timestamp: 11500, agent: 'sre', type: 'tool-result', toolName: 'check_service_health', result: 'api-gateway: HEALTHY | auth-service: HEALTHY | db-primary: HEALTHY (latency 45ms) | cache-redis: HEALTHY | worker-pool: HEALTHY' },
        { id: 'e12', timestamp: 12500, agent: 'sre', type: 'model-response', contentPreview: 'All 5 services HEALTHY. db-primary recovered — latency back to normal (45ms). Checkpoint completed. No action needed.' },
      ],
      totalDurationMs: 13500,
    },
  ],
}
