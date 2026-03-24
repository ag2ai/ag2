import type { ReplayData } from '../replay-types'

export const replay01: ReplayData = {
  example: {
    id: '01',
    title: 'Customer Support',
    category: 'Single Agent + Tools',
    description: 'One actor with domain tools handles customer inquiries — the simplest AG2 demo.',
    themes: ['autonomy'],
  },
  scenarios: [
    {
      id: 1,
      title: 'Order Status Inquiry',
      inputMessage: 'Hi, I ordered a laptop (order #ORD-7821) 5 days ago and it still says processing. Can you check what\'s going on?',
      agents: [
        { id: 'support', name: 'Support Agent', role: 'Customer support', tools: ['search_faq', 'lookup_order', 'check_inventory', 'process_refund', 'escalate_ticket'], color: 'emerald', x: 50, y: 50 },
      ],
      events: [
        { id: 'e1', timestamp: 0, agent: 'support', type: 'tool-call', toolName: 'lookup_order', args: 'order_id="ORD-7821"' },
        { id: 'e2', timestamp: 1200, agent: 'support', type: 'tool-result', toolName: 'lookup_order', result: 'Order: ORD-7821 | Product: UltraBook Pro 16-inch | Status: PROCESSING | Est. Delivery: 3 days | Notes: Awaiting stock from warehouse' },
        { id: 'e3', timestamp: 2800, agent: 'support', type: 'tool-call', toolName: 'search_faq', args: 'query="shipping time processing"' },
        { id: 'e4', timestamp: 3800, agent: 'support', type: 'tool-result', toolName: 'search_faq', result: 'Standard shipping: 5-7 business days. Express: 2-3 days. Free shipping on orders over $50.' },
        { id: 'e5', timestamp: 5500, agent: 'support', type: 'model-response', contentPreview: 'Hi! I\'ve looked into your order ORD-7821 for the UltraBook Pro 16-inch Laptop ($1,299.99). It\'s currently in PROCESSING status — the warehouse is preparing it for shipment and it should ship within 24 hours. Your estimated delivery date is in 3 days. I\'ll make sure you receive a tracking number via email once it ships!' },
      ],
      totalDurationMs: 6500,
    },
    {
      id: 2,
      title: 'Refund Request',
      inputMessage: 'I received my wireless headphones (order #ORD-6234) yesterday but they won\'t connect to any device. I want a refund.',
      agents: [
        { id: 'support', name: 'Support Agent', role: 'Customer support', tools: ['search_faq', 'lookup_order', 'check_inventory', 'process_refund', 'escalate_ticket'], color: 'emerald', x: 50, y: 50 },
      ],
      events: [
        { id: 'e1', timestamp: 0, agent: 'support', type: 'tool-call', toolName: 'lookup_order', args: 'order_id="ORD-6234"' },
        { id: 'e2', timestamp: 1000, agent: 'support', type: 'tool-result', toolName: 'lookup_order', result: 'Order: ORD-6234 | Product: ProSound Wireless Headphones XR-500 | Price: $149.99 | Status: DELIVERED | Tracking: FX-98234571-US' },
        { id: 'e3', timestamp: 2200, agent: 'support', type: 'tool-call', toolName: 'process_refund', args: 'order_id="ORD-6234", reason="Defective - won\'t connect to any device"' },
        { id: 'e4', timestamp: 3200, agent: 'support', type: 'tool-result', toolName: 'process_refund', result: 'Refund ID: RFD-847291 | Amount: $149.99 | Status: APPROVED — refund in 5-7 business days' },
        { id: 'e5', timestamp: 5000, agent: 'support', type: 'model-response', contentPreview: 'I\'m sorry to hear about the connectivity issues with your ProSound Wireless Headphones. I\'ve confirmed your order ORD-6234 and processed a full refund of $149.99 (Refund ID: RFD-847291). The refund will appear in your account within 5-7 business days. A confirmation email has been sent.' },
      ],
      totalDurationMs: 6000,
    },
    {
      id: 3,
      title: 'Product Availability',
      inputMessage: 'Do you have the UltraBook Pro 16 in stock? I need it delivered by Friday. Also, what\'s your return policy for electronics?',
      agents: [
        { id: 'support', name: 'Support Agent', role: 'Customer support', tools: ['search_faq', 'lookup_order', 'check_inventory', 'process_refund', 'escalate_ticket'], color: 'emerald', x: 50, y: 50 },
      ],
      events: [
        { id: 'e1', timestamp: 0, agent: 'support', type: 'tool-call', toolName: 'check_inventory', args: 'product_name="UltraBook Pro 16"' },
        { id: 'e2', timestamp: 1100, agent: 'support', type: 'tool-result', toolName: 'check_inventory', result: 'Product: UltraBook Pro 16-inch Laptop | SKU: UBP-16-2026 | Price: $1,299.99 | Status: IN STOCK (23 units) | Ships Within: 1-2 business days' },
        { id: 'e3', timestamp: 2500, agent: 'support', type: 'tool-call', toolName: 'search_faq', args: 'query="return policy electronics"' },
        { id: 'e4', timestamp: 3400, agent: 'support', type: 'tool-result', toolName: 'search_faq', result: 'Electronics have a 15-day return window. Items must be in original packaging. 1-year manufacturer warranty included. Extended warranty available.' },
        { id: 'e5', timestamp: 5200, agent: 'support', type: 'model-response', contentPreview: 'Great news! The UltraBook Pro 16 is IN STOCK with 23 units available at $1,299.99. It ships within 1-2 business days from our West Coast DC, so Friday delivery is very likely with express shipping. For electronics, we offer a 15-day return window (items must be in original packaging) plus a 1-year manufacturer warranty with optional extended coverage at checkout.' },
      ],
      totalDurationMs: 6200,
    },
  ],
}
