import type { ReplayData } from '../replay-types'

export const replay04: ReplayData = {
  example: {
    id: '04',
    title: '911 Emergency',
    category: 'Distributed Network',
    description: 'Always-on 911 dispatch center across 4 servers. Priority triage, live EMS-hospital transport updates, and hot-adding a fire department at runtime.',
    themes: ['network', 'distributed'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Critical Highway Accident',
      inputMessage: 'EMERGENCY: Car accident on Highway 101 near exit 42. Driver trapped, severely injured. Southbound lanes blocked.',
      agents: [
        { id: 'dispatch', name: 'Dispatch', role: '911 coordinator — triages severity, routes to services', tools: ['log_emergency', 'discover_agents', 'delegate_to'], server: 'dispatch-server', color: 'blue', x: 15, y: 50 },
        { id: 'ems', name: 'EMS', role: 'Ambulance dispatch and patient care', tools: ['dispatch_ambulance', 'assess_patient', 'update_patient_status'], server: 'medical-server', color: 'emerald', x: 65, y: 20 },
        { id: 'hospital', name: 'Hospital', role: 'ER preparation and specialist assignment', tools: ['check_er_capacity', 'prepare_trauma_bay', 'assign_specialist'], server: 'medical-server', color: 'emerald', x: 88, y: 20 },
        { id: 'police', name: 'Police', role: 'Traffic control and scene security', tools: ['dispatch_patrol_unit', 'setup_traffic_control', 'file_incident_report'], server: 'police-server', color: 'amber', x: 65, y: 75 },
      ],
      servers: [
        { id: 'dispatch-server', label: 'Dispatch Center :8900', agents: ['dispatch'], color: 'blue', x: 0, y: 20, w: 30, h: 60 },
        { id: 'medical-server', label: 'Medical Server :8901', agents: ['ems', 'hospital'], color: 'emerald', x: 48, y: 0, w: 52, h: 40 },
        { id: 'police-server', label: 'Police Server :8902', agents: ['police'], color: 'amber', x: 48, y: 52, w: 52, h: 45 },
      ],
      serverConnections: [
        { from: 'dispatch-server', to: 'medical-server', label: 'HTTP' },
        { from: 'dispatch-server', to: 'police-server', label: 'HTTP' },
      ],
      events: [
        // Dispatch receives emergency via POST /emergency
        { id: 'e01', timestamp: 0, agent: 'dispatch', type: 'tool-call', toolName: 'log_emergency', args: 'caller="Maria Torres", location="Highway 101 exit 42", severity="critical"' },
        { id: 'e02', timestamp: 1000, agent: 'dispatch', type: 'tool-result', toolName: 'log_emergency', result: 'Incident INC-20260324-4271 created. Severity: CRITICAL. Time: 14:32:07' },

        // Dispatch discovers available services
        { id: 'e03', timestamp: 1800, agent: 'dispatch', type: 'tool-call', toolName: 'discover_agents', args: '' },
        { id: 'e04', timestamp: 2200, agent: 'dispatch', type: 'tool-result', toolName: 'discover_agents', result: '- ems [medical, ambulance, patient-care]\n- hospital [emergency-room, trauma, specialists]\n- police [traffic, security, investigation]' },

        // Dispatch delegates to EMS (cross-server HTTP)
        { id: 'e05', timestamp: 3000, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'ems', channel: 'http', taskPreview: 'CRITICAL: Car accident Highway 101 exit 42. Driver trapped, severely injured, visible blood. Dispatch ambulance and assess patient immediately.' },

        // EMS runs tools
        { id: 'e06', timestamp: 4500, agent: 'ems', type: 'tool-call', toolName: 'dispatch_ambulance', args: 'location="Highway 101 exit 42", priority="critical"' },
        { id: 'e07', timestamp: 5500, agent: 'ems', type: 'tool-result', toolName: 'dispatch_ambulance', result: 'Ambulance AMB-472 dispatched. Priority: critical, ETA: 6 min. Crew: 2 paramedics + 1 EMT, Equipment: ALS' },
        { id: 'e08', timestamp: 6500, agent: 'ems', type: 'tool-call', toolName: 'assess_patient', args: 'symptoms="trapped, severe injuries, visible blood, barely moving", mechanism_of_injury="sedan flipped after hitting median barrier"' },
        { id: 'e09', timestamp: 7500, agent: 'ems', type: 'tool-result', toolName: 'assess_patient', result: 'Triage: RED (Immediate). Vitals: BP 90/60, HR 120, SpO2 92%. Suspected internal bleeding, possible fractures. Recommendation: Immediate transport to Level 1 trauma center' },

        // EMS delegates to Hospital — initial alert (local, same server)
        { id: 'e10', timestamp: 8500, agent: 'ems', type: 'delegation-request', source: 'ems', target: 'hospital', channel: 'local', taskPreview: 'INITIAL ALERT: Incoming trauma patient from Highway 101 accident. Triage RED. BP 90/60, HR 120, SpO2 92%. Suspected internal bleeding. ETA 15 min. Prepare trauma bay.' },
        { id: 'e11', timestamp: 9500, agent: 'hospital', type: 'tool-call', toolName: 'check_er_capacity', args: 'department="trauma"' },
        { id: 'e12', timestamp: 10200, agent: 'hospital', type: 'tool-result', toolName: 'check_er_capacity', result: 'ER [TRAUMA]: 3/12 beds, 2 trauma bays open. Status: ACCEPTING' },
        { id: 'e13', timestamp: 11000, agent: 'hospital', type: 'tool-call', toolName: 'prepare_trauma_bay', args: 'patient_info="trapped driver, internal bleeding, fractures", eta_minutes=15' },
        { id: 'e14', timestamp: 12000, agent: 'hospital', type: 'tool-result', toolName: 'prepare_trauma_bay', result: 'Trauma Bay 2 prepared. Blood on standby, CT reserved, surgical team notified.' },
        { id: 'e15', timestamp: 12800, agent: 'hospital', type: 'tool-call', toolName: 'assign_specialist', args: 'specialty="trauma-surgery", urgency="stat"' },
        { id: 'e16', timestamp: 13500, agent: 'hospital', type: 'tool-result', toolName: 'assign_specialist', result: 'Dr. Chen (trauma-surgery) assigned, urgency: stat, ETA: 3 min' },
        { id: 'e17', timestamp: 14200, agent: 'hospital', type: 'delegation-result', source: 'ems', target: 'hospital', resultPreview: 'Trauma Bay 2 ready. Dr. Chen (trauma-surgery) standing by. Blood on standby, CT reserved.' },

        // EMS updates patient status during transport
        { id: 'e18', timestamp: 15000, agent: 'ems', type: 'tool-call', toolName: 'update_patient_status', args: 'status="En route, IV established, oxygen administered", vitals="BP 95/65, HR 110, SpO2 95%"' },
        { id: 'e19', timestamp: 15800, agent: 'ems', type: 'tool-result', toolName: 'update_patient_status', result: 'Patient Update [14:34:52]: En route, IV established, oxygen administered. Vitals: BP 95/65, HR 110, SpO2 95%' },

        // EMS delegates to Hospital AGAIN — transport update (live communication)
        { id: 'e20', timestamp: 16500, agent: 'ems', type: 'delegation-request', source: 'ems', target: 'hospital', channel: 'local', taskPreview: 'TRANSPORT UPDATE: Patient stabilizing. BP 95/65, HR 110, SpO2 95%. IV + O2 in place. Revised ETA 8 min. Confirm readiness.' },
        { id: 'e21', timestamp: 17500, agent: 'hospital', type: 'tool-call', toolName: 'assign_specialist', args: 'specialty="orthopedics", urgency="urgent"' },
        { id: 'e22', timestamp: 18300, agent: 'hospital', type: 'tool-result', toolName: 'assign_specialist', result: 'Dr. Park (orthopedics) assigned, urgency: urgent, ETA: 5 min' },
        { id: 'e23', timestamp: 19000, agent: 'hospital', type: 'delegation-result', source: 'ems', target: 'hospital', resultPreview: 'Ready for arrival. Trauma Bay 2 active. Dr. Chen + Dr. Park standing by. Revised ETA acknowledged.' },

        // EMS result returns to dispatch
        { id: 'e24', timestamp: 20000, agent: 'ems', type: 'delegation-result', source: 'dispatch', target: 'ems', resultPreview: 'AMB-472 en route. Patient stabilizing (BP 95/65). Trauma Bay 2 ready with Dr. Chen + Dr. Park. ETA 8 min.' },

        // Meanwhile: dispatch delegates to police (cross-server HTTP, parallel)
        { id: 'e25', timestamp: 3500, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'police', channel: 'http', taskPreview: 'CRITICAL: Highway 101 southbound at exit 42 fully blocked. Vehicle accident. Set up traffic control and secure scene.' },
        { id: 'e26', timestamp: 5000, agent: 'police', type: 'tool-call', toolName: 'dispatch_patrol_unit', args: 'location="Highway 101 exit 42", unit_type="traffic"' },
        { id: 'e27', timestamp: 5800, agent: 'police', type: 'tool-result', toolName: 'dispatch_patrol_unit', result: 'Police traffic UNIT-34 dispatched. ETA: 5 min, Officers: 2' },
        { id: 'e28', timestamp: 6800, agent: 'police', type: 'tool-call', toolName: 'setup_traffic_control', args: 'location="Highway 101 exit 42", action="full-closure", lanes_affected="southbound"' },
        { id: 'e29', timestamp: 7600, agent: 'police', type: 'tool-result', toolName: 'setup_traffic_control', result: 'Traffic Control at Highway 101 exit 42: full-closure, lanes: southbound. Barriers deployed, nav advisory issued.' },
        { id: 'e30', timestamp: 8500, agent: 'police', type: 'tool-call', toolName: 'file_incident_report', args: 'incident_type="accident", details="Single vehicle rollover, driver trapped"' },
        { id: 'e31', timestamp: 9300, agent: 'police', type: 'tool-result', toolName: 'file_incident_report', result: 'Report RPT-47291 filed: accident. Single vehicle rollover, driver trapped' },
        { id: 'e32', timestamp: 10000, agent: 'police', type: 'delegation-result', source: 'dispatch', target: 'police', resultPreview: 'Scene secured. UNIT-34 on site. Southbound lanes closed, traffic diverted. Report RPT-47291 filed.' },

        // Final dispatch summary
        { id: 'e33', timestamp: 21000, agent: 'dispatch', type: 'model-response', contentPreview: 'INCIDENT INC-20260324-4271 — CRITICAL. Highway 101 exit 42 vehicle accident.\n\nEMS: AMB-472 en route, patient stabilizing (BP 95/65, SpO2 95%). Trauma Bay 2 ready at Regional Medical with Dr. Chen (trauma-surgery) + Dr. Park (orthopedics).\n\nPolice: UNIT-34 on scene, southbound lanes closed, detour active. Report RPT-47291 filed.\n\nTwo live EMS→Hospital updates completed during transport.' },
      ],
      totalDurationMs: 22000,
    },
    {
      id: 2,
      title: 'Minor Fender Bender',
      inputMessage: 'EMERGENCY: Low-speed fender bender at Main St and 2nd Ave. Bumper damage only. No injuries, drivers exchanging info.',
      agents: [
        { id: 'dispatch', name: 'Dispatch', role: '911 coordinator — triages severity, routes to services', tools: ['log_emergency', 'discover_agents', 'delegate_to'], server: 'dispatch-server', color: 'blue', x: 15, y: 50 },
        { id: 'ems', name: 'EMS', role: 'Ambulance dispatch and patient care', tools: ['dispatch_ambulance', 'assess_patient', 'update_patient_status'], server: 'medical-server', color: 'emerald', x: 65, y: 20 },
        { id: 'hospital', name: 'Hospital', role: 'ER preparation and specialist assignment', tools: ['check_er_capacity', 'prepare_trauma_bay', 'assign_specialist'], server: 'medical-server', color: 'emerald', x: 88, y: 20 },
        { id: 'police', name: 'Police', role: 'Traffic control and scene security', tools: ['dispatch_patrol_unit', 'setup_traffic_control', 'file_incident_report'], server: 'police-server', color: 'amber', x: 65, y: 75 },
      ],
      servers: [
        { id: 'dispatch-server', label: 'Dispatch Center :8900', agents: ['dispatch'], color: 'blue', x: 0, y: 20, w: 30, h: 60 },
        { id: 'medical-server', label: 'Medical Server :8901', agents: ['ems', 'hospital'], color: 'emerald', x: 48, y: 0, w: 52, h: 40 },
        { id: 'police-server', label: 'Police Server :8902', agents: ['police'], color: 'amber', x: 48, y: 52, w: 52, h: 45 },
      ],
      serverConnections: [
        { from: 'dispatch-server', to: 'medical-server', label: 'HTTP' },
        { from: 'dispatch-server', to: 'police-server', label: 'HTTP' },
      ],
      events: [
        // Dispatch logs as MINOR
        { id: 'e01', timestamp: 0, agent: 'dispatch', type: 'tool-call', toolName: 'log_emergency', args: 'caller="Anonymous", location="Main St & 2nd Ave", severity="minor"' },
        { id: 'e02', timestamp: 800, agent: 'dispatch', type: 'tool-result', toolName: 'log_emergency', result: 'Incident INC-20260324-8103 created. Severity: MINOR. Time: 15:45:12' },

        { id: 'e03', timestamp: 1500, agent: 'dispatch', type: 'tool-call', toolName: 'discover_agents', args: '' },
        { id: 'e04', timestamp: 1900, agent: 'dispatch', type: 'tool-result', toolName: 'discover_agents', result: '- ems [medical, ambulance, patient-care]\n- hospital [emergency-room, trauma, specialists]\n- police [traffic, security, investigation]' },

        // MINOR: only police, no EMS needed
        { id: 'e05', timestamp: 2800, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'police', channel: 'http', taskPreview: 'MINOR: Fender bender at Main St & 2nd Ave. No injuries. Bumper damage only. File report and manage traffic.' },
        { id: 'e06', timestamp: 4000, agent: 'police', type: 'tool-call', toolName: 'dispatch_patrol_unit', args: 'location="Main St & 2nd Ave", unit_type="patrol"' },
        { id: 'e07', timestamp: 4800, agent: 'police', type: 'tool-result', toolName: 'dispatch_patrol_unit', result: 'Police patrol UNIT-18 dispatched. ETA: 7 min, Officers: 2' },
        { id: 'e08', timestamp: 5800, agent: 'police', type: 'tool-call', toolName: 'file_incident_report', args: 'incident_type="accident", details="Low-speed fender bender, no injuries, property damage only"' },
        { id: 'e09', timestamp: 6600, agent: 'police', type: 'tool-result', toolName: 'file_incident_report', result: 'Report RPT-51034 filed: accident. Low-speed fender bender, no injuries, property damage only' },
        { id: 'e10', timestamp: 7500, agent: 'police', type: 'delegation-result', source: 'dispatch', target: 'police', resultPreview: 'UNIT-18 en route. Report RPT-51034 filed. No traffic control needed — minor lane obstruction.' },

        // Dispatch summary — no EMS was dispatched
        { id: 'e11', timestamp: 8500, agent: 'dispatch', type: 'model-response', contentPreview: 'INCIDENT INC-20260324-8103 — MINOR. Fender bender at Main St & 2nd Ave.\n\nNo injuries reported. EMS not dispatched.\nPolice: UNIT-18 en route for report. RPT-51034 filed.\n\nMinimal response appropriate for severity level.' },
      ],
      totalDurationMs: 9500,
    },
    {
      id: 3,
      title: 'Warehouse Fire (with Fire Dept)',
      inputMessage: 'EMERGENCY: Large fire at Henderson Warehouse on Industrial Blvd. Loading dock engulfed. Two workers unaccounted for. Chemical storage at risk. Explosions heard.',
      agents: [
        { id: 'dispatch', name: 'Dispatch', role: '911 coordinator — triages severity, routes to services', tools: ['log_emergency', 'discover_agents', 'delegate_to'], server: 'dispatch-server', color: 'blue', x: 8, y: 50 },
        { id: 'ems', name: 'EMS', role: 'Ambulance dispatch and patient care', tools: ['dispatch_ambulance', 'assess_patient', 'update_patient_status'], server: 'medical-server', color: 'emerald', x: 52, y: 15 },
        { id: 'hospital', name: 'Hospital', role: 'ER preparation and specialist assignment', tools: ['check_er_capacity', 'prepare_trauma_bay', 'assign_specialist'], server: 'medical-server', color: 'emerald', x: 75, y: 15 },
        { id: 'police', name: 'Police', role: 'Traffic control and scene security', tools: ['dispatch_patrol_unit', 'setup_traffic_control', 'file_incident_report'], server: 'police-server', color: 'amber', x: 52, y: 85 },
        { id: 'fire', name: 'Fire Chief', role: 'Fire suppression, hazmat, and rescue', tools: ['dispatch_fire_engine', 'assess_fire_hazard', 'establish_perimeter'], server: 'fire-server', color: 'red', x: 75, y: 50 },
      ],
      servers: [
        { id: 'dispatch-server', label: 'Dispatch Center :8900', agents: ['dispatch'], color: 'blue', x: 0, y: 25, w: 22, h: 50 },
        { id: 'medical-server', label: 'Medical Server :8901', agents: ['ems', 'hospital'], color: 'emerald', x: 38, y: 0, w: 52, h: 30 },
        { id: 'police-server', label: 'Police Server :8902', agents: ['police'], color: 'amber', x: 38, y: 68, w: 25, h: 30 },
        { id: 'fire-server', label: 'Fire Dept :8903', agents: ['fire'], color: 'red', x: 65, y: 35, w: 25, h: 30 },
      ],
      serverConnections: [
        { from: 'dispatch-server', to: 'medical-server', label: 'HTTP' },
        { from: 'dispatch-server', to: 'police-server', label: 'HTTP' },
        { from: 'dispatch-server', to: 'fire-server', label: 'HTTP (hot-connected)' },
      ],
      events: [
        // Dispatch logs CRITICAL fire
        { id: 'e01', timestamp: 0, agent: 'dispatch', type: 'tool-call', toolName: 'log_emergency', args: 'caller="Multiple callers", location="Henderson Warehouse, Industrial Blvd", severity="critical"' },
        { id: 'e02', timestamp: 1000, agent: 'dispatch', type: 'tool-result', toolName: 'log_emergency', result: 'Incident INC-20260324-6619 created. Severity: CRITICAL. Time: 16:12:33' },

        { id: 'e03', timestamp: 1800, agent: 'dispatch', type: 'tool-call', toolName: 'discover_agents', args: '' },
        { id: 'e04', timestamp: 2200, agent: 'dispatch', type: 'tool-result', toolName: 'discover_agents', result: '- ems [medical, ambulance, patient-care]\n- hospital [emergency-room, trauma, specialists]\n- police [traffic, security, investigation]\n- fire [fire, hazmat, rescue]' },

        // Dispatch delegates to fire (hot-connected server)
        { id: 'e05', timestamp: 3000, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'fire', channel: 'http', taskPreview: 'CRITICAL: Large structure fire at Henderson Warehouse, Industrial Blvd. Loading dock engulfed, chemical storage at risk. Two workers missing. Explosions reported.' },
        { id: 'e06', timestamp: 4500, agent: 'fire', type: 'tool-call', toolName: 'dispatch_fire_engine', args: 'location="Henderson Warehouse, Industrial Blvd", engine_type="ladder"' },
        { id: 'e07', timestamp: 5300, agent: 'fire', type: 'tool-result', toolName: 'dispatch_fire_engine', result: 'Fire Engine ENG-312 (ladder) dispatched. ETA: 5 min, Crew: 6 firefighters' },
        { id: 'e08', timestamp: 6000, agent: 'fire', type: 'tool-call', toolName: 'dispatch_fire_engine', args: 'location="Henderson Warehouse, Industrial Blvd", engine_type="hazmat"' },
        { id: 'e09', timestamp: 6800, agent: 'fire', type: 'tool-result', toolName: 'dispatch_fire_engine', result: 'Fire Engine ENG-508 (hazmat) dispatched. ETA: 9 min, Crew: 4 firefighters' },
        { id: 'e10', timestamp: 7500, agent: 'fire', type: 'tool-call', toolName: 'assess_fire_hazard', args: 'location="Henderson Warehouse", situation="loading dock engulfed, chemical storage nearby, explosions"' },
        { id: 'e11', timestamp: 8500, agent: 'fire', type: 'tool-result', toolName: 'assess_fire_hazard', result: 'Risk Level: EXTREME. Wind: SW at 18 mph. 5 structures within hazard radius. Establish 200ft perimeter.' },
        { id: 'e12', timestamp: 9200, agent: 'fire', type: 'tool-call', toolName: 'establish_perimeter', args: 'location="Henderson Warehouse", radius_feet=200' },
        { id: 'e13', timestamp: 10000, agent: 'fire', type: 'tool-result', toolName: 'establish_perimeter', result: 'Safety perimeter established. Radius: 200 ft, Access points: 2. Evacuation: 35 civilians cleared' },
        { id: 'e14', timestamp: 10800, agent: 'fire', type: 'delegation-result', source: 'dispatch', target: 'fire', resultPreview: 'ENG-312 (ladder) + ENG-508 (hazmat) dispatched. 200ft perimeter established. 35 civilians evacuated. EXTREME risk — search for 2 missing workers underway.' },

        // Dispatch delegates to EMS (parallel)
        { id: 'e15', timestamp: 3500, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'ems', channel: 'http', taskPreview: 'CRITICAL: Warehouse fire. Two workers unaccounted for, potential burn/smoke inhalation injuries. Stage ambulance at perimeter.' },
        { id: 'e16', timestamp: 5000, agent: 'ems', type: 'tool-call', toolName: 'dispatch_ambulance', args: 'location="Henderson Warehouse, Industrial Blvd", priority="critical"' },
        { id: 'e17', timestamp: 5800, agent: 'ems', type: 'tool-result', toolName: 'dispatch_ambulance', result: 'Ambulance AMB-219 dispatched. Priority: critical, ETA: 7 min. Crew: 2 paramedics + 1 EMT, Equipment: ALS' },

        // EMS delegates to hospital (local)
        { id: 'e18', timestamp: 6800, agent: 'ems', type: 'delegation-request', source: 'ems', target: 'hospital', channel: 'local', taskPreview: 'INITIAL ALERT: Standby for potential burn/smoke inhalation victims from warehouse fire. Up to 2 patients. ETA unknown pending rescue.' },
        { id: 'e19', timestamp: 7800, agent: 'hospital', type: 'tool-call', toolName: 'check_er_capacity', args: 'department="trauma"' },
        { id: 'e20', timestamp: 8400, agent: 'hospital', type: 'tool-result', toolName: 'check_er_capacity', result: 'ER [TRAUMA]: 4/12 beds, 3 trauma bays open. Status: ACCEPTING' },
        { id: 'e21', timestamp: 9200, agent: 'hospital', type: 'tool-call', toolName: 'prepare_trauma_bay', args: 'patient_info="potential burn/smoke inhalation victims x2", eta_minutes=20' },
        { id: 'e22', timestamp: 10000, agent: 'hospital', type: 'tool-result', toolName: 'prepare_trauma_bay', result: 'Trauma Bay 1 prepared. Blood on standby, CT reserved, surgical team notified.' },
        { id: 'e23', timestamp: 10800, agent: 'hospital', type: 'delegation-result', source: 'ems', target: 'hospital', resultPreview: 'Trauma Bay 1 ready for burn patients. 3 bays available total. Standing by.' },
        { id: 'e24', timestamp: 11500, agent: 'ems', type: 'delegation-result', source: 'dispatch', target: 'ems', resultPreview: 'AMB-219 staged at perimeter. Hospital Trauma Bay 1 prepped for burn victims. Awaiting rescue outcome.' },

        // Dispatch delegates to police (parallel)
        { id: 'e25', timestamp: 4000, agent: 'dispatch', type: 'delegation-request', source: 'dispatch', target: 'police', channel: 'http', taskPreview: 'CRITICAL: Warehouse fire at Henderson Warehouse, Industrial Blvd. Secure area, block Industrial Blvd, assist evacuation.' },
        { id: 'e26', timestamp: 5500, agent: 'police', type: 'tool-call', toolName: 'dispatch_patrol_unit', args: 'location="Henderson Warehouse, Industrial Blvd", unit_type="traffic"' },
        { id: 'e27', timestamp: 6300, agent: 'police', type: 'tool-result', toolName: 'dispatch_patrol_unit', result: 'Police traffic UNIT-22 dispatched. ETA: 4 min, Officers: 2' },
        { id: 'e28', timestamp: 7000, agent: 'police', type: 'tool-call', toolName: 'setup_traffic_control', args: 'location="Industrial Blvd at Henderson", action="full-closure", lanes_affected="all"' },
        { id: 'e29', timestamp: 7800, agent: 'police', type: 'tool-result', toolName: 'setup_traffic_control', result: 'Traffic Control at Industrial Blvd at Henderson: full-closure, lanes: all. Barriers deployed, nav advisory issued.' },
        { id: 'e30', timestamp: 8500, agent: 'police', type: 'tool-call', toolName: 'file_incident_report', args: 'incident_type="hazard", details="Structure fire with explosions, chemical risk, 2 missing persons"' },
        { id: 'e31', timestamp: 9300, agent: 'police', type: 'tool-result', toolName: 'file_incident_report', result: 'Report RPT-62108 filed: hazard. Structure fire with explosions, chemical risk, 2 missing persons' },
        { id: 'e32', timestamp: 10000, agent: 'police', type: 'delegation-result', source: 'dispatch', target: 'police', resultPreview: 'Industrial Blvd closed. UNIT-22 assisting evacuation. Report RPT-62108 filed.' },

        // Final dispatch summary
        { id: 'e33', timestamp: 12000, agent: 'dispatch', type: 'model-response', contentPreview: 'INCIDENT INC-20260324-6619 — CRITICAL. Warehouse fire at Henderson Warehouse.\n\nFire: ENG-312 (ladder) + ENG-508 (hazmat) on scene. 200ft perimeter, 35 evacuated. Search for 2 missing workers active.\nEMS: AMB-219 staged. Trauma Bay 1 ready for burn victims.\nPolice: Industrial Blvd closed. UNIT-22 on scene. RPT-62108 filed.\n\nAll 3 services coordinated. Fire department hot-connected at runtime.' },
      ],
      totalDurationMs: 13000,
    },
  ],
}
