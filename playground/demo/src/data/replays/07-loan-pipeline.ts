import type { ReplayData } from '../replay-types'

export const replay07: ReplayData = {
  example: {
    id: '07',
    title: 'Loan Pipeline',
    category: 'Human-in-the-Loop',
    description: 'Three agents process loan applications. An ApprovalGate plugin pauses for human approval.',
    themes: ['network', 'autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Strong Applicant',
      inputMessage: 'New loan application: Sarah Chen, requesting $350,000 mortgage. Income: $185,000/yr. Credit score: 782. DTI: 28%.',
      agents: [
        { id: 'intake', name: 'Intake', role: 'Application intake', tools: ['validate_application', 'check_completeness', 'calculate_dti'], color: 'cyan', x: 20, y: 50 },
        { id: 'credit', name: 'Credit Analyst', role: 'Credit assessment', tools: ['pull_credit_report', 'analyze_debt_history', 'score_risk'], color: 'blue', x: 50, y: 50 },
        { id: 'underwriter', name: 'Underwriter', role: 'Final decision', tools: ['review_package', 'set_terms', 'issue_decision'], color: 'emerald', x: 80, y: 50 },
      ],
      events: [
        { id: 'e01', timestamp: 0, agent: 'intake', type: 'tool-call', toolName: 'validate_application', args: 'applicant="Sarah Chen", amount=350000' },
        { id: 'e02', timestamp: 1000, agent: 'intake', type: 'tool-result', toolName: 'validate_application', result: 'Application VALID. All required fields present. SSN verified. Employment verified: Senior Engineer at TechCorp (6 years).' },
        { id: 'e03', timestamp: 2000, agent: 'intake', type: 'tool-call', toolName: 'calculate_dti', args: 'income=185000, existing_debt=2100, new_payment=1850' },
        { id: 'e04', timestamp: 2800, agent: 'intake', type: 'tool-result', toolName: 'calculate_dti', result: 'DTI: 25.6% (front-end: 12.0%, back-end: 25.6%). Well below 36% threshold. EXCELLENT.' },
        { id: 'e05', timestamp: 3800, agent: 'intake', type: 'delegation-request', source: 'intake', target: 'credit', channel: 'local', taskPreview: 'Validated application for Sarah Chen. $350K mortgage, income $185K, DTI 25.6%. Proceed with credit analysis.' },
        { id: 'e06', timestamp: 5000, agent: 'credit', type: 'tool-call', toolName: 'pull_credit_report', args: 'applicant="Sarah Chen"' },
        { id: 'e07', timestamp: 6000, agent: 'credit', type: 'tool-result', toolName: 'pull_credit_report', result: 'Credit Score: 782 (Excellent). Accounts: 8 open, 0 delinquent. Oldest account: 12 years. Utilization: 14%. No collections, no bankruptcies.' },
        { id: 'e08', timestamp: 7000, agent: 'credit', type: 'tool-call', toolName: 'score_risk', args: 'credit_score=782, dti=25.6, loan_amount=350000' },
        { id: 'e09', timestamp: 8000, agent: 'credit', type: 'tool-result', toolName: 'score_risk', result: 'Risk Score: A+ (LOW RISK). LTV: 80%. Recommendation: APPROVE at preferred rate.' },
        // ApprovalGate pauses here for human approval
        { id: 'e10', timestamp: 9000, agent: 'credit', type: 'delegation-request', source: 'credit', target: 'underwriter', channel: 'local', taskPreview: '[APPROVAL GATE] Credit analysis complete for Sarah Chen. Score: 782, Risk: A+. Awaiting human approval to proceed to underwriting.' },
        { id: 'e11', timestamp: 11000, agent: 'underwriter', type: 'tool-call', toolName: 'review_package', args: 'applicant="Sarah Chen", risk_grade="A+"' },
        { id: 'e12', timestamp: 12000, agent: 'underwriter', type: 'tool-result', toolName: 'review_package', result: 'Package review: STRONG. Income stable (6yr tenure), excellent credit, low DTI, adequate reserves (8 months). No red flags.' },
        { id: 'e13', timestamp: 13000, agent: 'underwriter', type: 'tool-call', toolName: 'set_terms', args: 'rate="6.25%", term="30yr fixed", ltv=80' },
        { id: 'e14', timestamp: 13800, agent: 'underwriter', type: 'tool-result', toolName: 'set_terms', result: 'Terms set: 6.25% fixed, 30-year. Monthly payment: $2,155. Total interest: $425,800. Rate lock: 60 days.' },
        { id: 'e15', timestamp: 14800, agent: 'underwriter', type: 'tool-call', toolName: 'issue_decision', args: 'decision="APPROVED", applicant="Sarah Chen"' },
        { id: 'e16', timestamp: 15600, agent: 'underwriter', type: 'tool-result', toolName: 'issue_decision', result: 'DECISION: APPROVED. Loan ID: LN-20260324-0891. Commitment letter generated. Closing scheduled.' },
        { id: 'e17', timestamp: 16500, agent: 'underwriter', type: 'delegation-result', source: 'credit', target: 'underwriter', resultPreview: 'APPROVED at 6.25% fixed, 30yr. Loan ID: LN-20260324-0891.' },
        { id: 'e18', timestamp: 17000, agent: 'credit', type: 'delegation-result', source: 'intake', target: 'credit', resultPreview: 'Loan approved. $350K at 6.25% fixed. Monthly: $2,155.' },
        { id: 'e19', timestamp: 18000, agent: 'intake', type: 'model-response', contentPreview: 'Loan application for Sarah Chen: APPROVED. $350,000 mortgage at 6.25% fixed (30yr), monthly payment $2,155. Credit score 782, DTI 25.6%, Risk grade A+. Loan ID: LN-20260324-0891. Commitment letter generated, closing to be scheduled.' },
      ],
      totalDurationMs: 19000,
    },
  ],
}
