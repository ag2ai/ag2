import type { ReplayData } from '../replay-types'

export const replay05: ReplayData = {
  example: {
    id: '05',
    title: 'Smart Building',
    category: 'Distributed + Scheduled',
    description: 'HVAC, Energy, Security, and Maintenance agents across two servers with autonomous scheduling.',
    themes: ['network', 'distributed', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Security Breach Response',
      inputMessage: 'ALERT: Unauthorized access detected at server room door SR-01 at 22:47. Camera shows unrecognized individual. Building is in after-hours mode.',
      agents: [
        { id: 'security', name: 'Security', role: 'Security monitoring', tools: ['check_cameras', 'lock_zone', 'trigger_alarm', 'notify_authorities'], server: 'operations-server', color: 'rose', x: 65, y: 25 },
        { id: 'maintenance', name: 'Maintenance', role: 'Building systems', tools: ['check_door_status', 'override_lock', 'run_diagnostics'], server: 'operations-server', color: 'amber', x: 85, y: 25 },
        { id: 'hvac', name: 'HVAC', role: 'Climate control', tools: ['get_zone_temp', 'set_zone_temp', 'check_air_quality'], server: 'climate-server', color: 'cyan', x: 65, y: 75 },
        { id: 'energy', name: 'Energy', role: 'Power management', tools: ['get_power_usage', 'switch_mode', 'activate_emergency_lighting'], server: 'climate-server', color: 'emerald', x: 85, y: 75 },
      ],
      servers: [
        { id: 'controller', label: 'Building Controller :9000', agents: [], color: 'blue', x: 0, y: 20, w: 30, h: 60 },
        { id: 'operations-server', label: 'Operations Server :9001', agents: ['security', 'maintenance'], color: 'amber', x: 48, y: 0, w: 52, h: 45 },
        { id: 'climate-server', label: 'Climate Server :9002', agents: ['hvac', 'energy'], color: 'cyan', x: 48, y: 52, w: 52, h: 45 },
      ],
      serverConnections: [
        { from: 'controller', to: 'operations-server', label: 'HttpChannel' },
        { from: 'controller', to: 'climate-server', label: 'HttpChannel' },
      ],
      events: [
        // Controller connects
        { id: 'e01', timestamp: 0, agent: 'security', type: 'hub-connect', targetServer: 'operations-server', agentsDiscovered: ['security', 'maintenance'] },
        { id: 'e02', timestamp: 600, agent: 'energy', type: 'hub-connect', targetServer: 'climate-server', agentsDiscovered: ['hvac', 'energy'] },

        // Security investigates (cross-server delegation from controller)
        { id: 'e03', timestamp: 1500, agent: 'security', type: 'tool-call', toolName: 'check_cameras', args: 'zone="server-room", time="22:47"' },
        { id: 'e04', timestamp: 2500, agent: 'security', type: 'tool-result', toolName: 'check_cameras', result: 'Camera SR-CAM-01: Unrecognized male, dark clothing, attempting badge reader. Camera SR-CAM-02: Same individual, carrying laptop bag.' },
        { id: 'e05', timestamp: 3500, agent: 'security', type: 'tool-call', toolName: 'lock_zone', args: 'zone="server-room", mode="emergency_lockdown"' },
        { id: 'e06', timestamp: 4300, agent: 'security', type: 'tool-result', toolName: 'lock_zone', result: 'Server room SR-01 LOCKED DOWN. All entry points sealed. Badge access disabled.' },
        { id: 'e07', timestamp: 5000, agent: 'security', type: 'tool-call', toolName: 'trigger_alarm', args: 'type="intrusion", zone="server-room"' },
        { id: 'e08', timestamp: 5800, agent: 'security', type: 'tool-result', toolName: 'trigger_alarm', result: 'Intrusion alarm activated for server room zone. Silent alert sent to security team.' },
        { id: 'e09', timestamp: 6500, agent: 'security', type: 'tool-call', toolName: 'notify_authorities', args: 'type="break-in attempt", location="Building A, Server Room SR-01"' },
        { id: 'e10', timestamp: 7500, agent: 'security', type: 'tool-result', toolName: 'notify_authorities', result: 'Local police notified. Case #PD-20260324-1147. ETA: 12 minutes.' },

        // Security delegates to Maintenance (local — same server)
        { id: 'e11', timestamp: 8500, agent: 'security', type: 'delegation-request', source: 'security', target: 'maintenance', channel: 'local', taskPreview: 'Verify server room door SR-01 lock integrity. Check if physical lock was tampered with. Run diagnostics on access control system.' },
        { id: 'e12', timestamp: 9500, agent: 'maintenance', type: 'tool-call', toolName: 'check_door_status', args: 'door="SR-01"' },
        { id: 'e13', timestamp: 10300, agent: 'maintenance', type: 'tool-result', toolName: 'check_door_status', result: 'Door SR-01: LOCKED (emergency). Electromagnetic lock: engaged. No physical tampering detected. Badge reader: 3 failed attempts logged.' },
        { id: 'e14', timestamp: 11000, agent: 'maintenance', type: 'tool-call', toolName: 'run_diagnostics', args: 'system="access-control"' },
        { id: 'e15', timestamp: 12000, agent: 'maintenance', type: 'tool-result', toolName: 'run_diagnostics', result: 'Access control system: NOMINAL. No firmware anomalies. Badge DB: no matching badge for failed attempts. Recommendation: physical inspection on next shift.' },
        { id: 'e16', timestamp: 13000, agent: 'maintenance', type: 'delegation-result', source: 'security', target: 'maintenance', resultPreview: 'Door SR-01 secure. No tampering detected. 3 failed badge attempts logged.' },

        // Cross-server: Security coordinates with Energy for emergency lighting
        { id: 'e17', timestamp: 9000, agent: 'security', type: 'delegation-request', source: 'security', target: 'energy', channel: 'http', taskPreview: 'Switch building to emergency lighting mode. Unauthorized access in server room — need full corridor visibility for security response.' },
        { id: 'e18', timestamp: 10500, agent: 'energy', type: 'tool-call', toolName: 'switch_mode', args: 'mode="emergency", zones=["server-room", "corridors", "lobby"]' },
        { id: 'e19', timestamp: 11500, agent: 'energy', type: 'tool-result', toolName: 'switch_mode', result: 'Emergency mode ACTIVE. Full lighting in server room corridor, all exits, lobby. Power usage up 2.1 kW.' },
        { id: 'e20', timestamp: 12200, agent: 'energy', type: 'tool-call', toolName: 'activate_emergency_lighting', args: 'floors=["2", "3"]' },
        { id: 'e21', timestamp: 13000, agent: 'energy', type: 'tool-result', toolName: 'activate_emergency_lighting', result: 'Emergency lighting active on floors 2-3. Exit signs at maximum brightness. Battery backup engaged.' },
        { id: 'e22', timestamp: 13800, agent: 'energy', type: 'delegation-result', source: 'security', target: 'energy', resultPreview: 'Emergency lighting active. Full visibility on affected floors.' },

        // Final response
        { id: 'e23', timestamp: 15000, agent: 'security', type: 'model-response', contentPreview: 'SECURITY BREACH RESPONSE COMPLETE. Server room SR-01: locked down, all entry sealed. Intrusion alarm active. Police notified (Case #PD-20260324-1147, ETA 12 min). Door integrity verified — no tampering, 3 failed badge attempts logged. Emergency lighting active on floors 2-3. All building systems in emergency mode. Awaiting police arrival.' },
      ],
      totalDurationMs: 16000,
    },
  ],
}
