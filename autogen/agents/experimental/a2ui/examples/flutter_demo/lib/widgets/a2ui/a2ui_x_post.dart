import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

/// Custom XPost (Twitter/X) component.
class A2UIXPost extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UIXPost({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final displayName = component['authorName'] as String? ?? component['displayName'] as String? ?? '';
    final handle = component['authorHandle'] as String? ?? component['handle'] as String? ?? '';
    final verified = component['verified'] as bool? ?? false;
    final body = component['body'] as String? ?? '';
    final likes = component['likes'] as int? ?? 0;
    final reposts = component['reposts'] as int? ?? 0;
    final replies = component['replies'] as int? ?? 0;
    final views = component['views'] as int? ?? 0;
    final mediaChild = component['mediaChild'] as String?;
    final children = mediaChild != null ? [mediaChild] : (component['children'] as List<dynamic>?)?.cast<String>() ?? [];

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        border: Border.all(color: const Color(0xFF334155)),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Row(
            children: [
              CircleAvatar(
                radius: 20,
                backgroundColor: const Color(0xFF334155),
                child: Text(
                  displayName.isNotEmpty ? displayName[0].toUpperCase() : '?',
                  style: const TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(displayName, style: const TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.w600, fontSize: 14)),
                      if (verified) ...[
                        const SizedBox(width: 4),
                        const Icon(Icons.verified, size: 16, color: Color(0xFF1D9BF0)),
                      ],
                    ],
                  ),
                  Text('@$handle', style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Body
          Text(body, style: const TextStyle(color: Color(0xFFE2E8F0), fontSize: 14)),
          // Media child
          for (final childId in children) ...[
            const SizedBox(height: 10),
            buildComponent(childId, components, surfaceId, state),
          ],
          const Divider(color: Color(0xFF334155), height: 20),
          // Engagement
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _engagementItem(Icons.chat_bubble_outline, '$replies'),
              _engagementItem(Icons.repeat, '$reposts'),
              _engagementItem(Icons.favorite_outline, '$likes'),
              _engagementItem(Icons.bar_chart, '$views'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _engagementItem(IconData icon, String count) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: const Color(0xFF94A3B8)),
        const SizedBox(width: 4),
        Text(count, style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
      ],
    );
  }
}
