import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

/// Custom LinkedInPost component — renders avatar row, body, hashtags, media, and engagement.
class A2UILinkedInPost extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UILinkedInPost({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final authorName = component['authorName'] as String? ?? '';
    final authorTitle = component['authorHeadline'] as String? ?? component['authorTitle'] as String? ?? '';
    final body = component['body'] as String? ?? '';
    final hashtags = component['hashtags'] is String
        ? (component['hashtags'] as String).split(' ').where((s) => s.isNotEmpty).toList()
        : (component['hashtags'] as List<dynamic>?)?.cast<String>() ?? [];
    final likes = component['likes'] as int? ?? 0;
    final comments = component['comments'] as int? ?? 0;
    final reposts = component['reposts'] as int? ?? 0;
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
          // Author row
          Row(
            children: [
              CircleAvatar(
                radius: 20,
                backgroundColor: const Color(0xFF334155),
                child: Text(
                  authorName.isNotEmpty ? authorName[0].toUpperCase() : '?',
                  style: const TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(authorName, style: const TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.w600, fontSize: 14)),
                  if (authorTitle.isNotEmpty)
                    Text(authorTitle, style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Body
          Text(body, style: const TextStyle(color: Color(0xFFE2E8F0), fontSize: 14)),
          if (hashtags.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              hashtags.map((h) => h.startsWith('#') ? h : '#$h').join(' '),
              style: const TextStyle(color: Color(0xFF818CF8), fontSize: 13),
            ),
          ],
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
              _engagementItem(Icons.thumb_up_outlined, '$likes'),
              _engagementItem(Icons.comment_outlined, '$comments'),
              _engagementItem(Icons.repeat, '$reposts'),
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
