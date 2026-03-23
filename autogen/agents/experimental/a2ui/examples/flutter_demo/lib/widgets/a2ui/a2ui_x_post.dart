import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

/// Custom XPost (Twitter/X) component — matches real X dark theme formatting.
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

  String _formatNumber(int n) {
    if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}K';
    return '$n';
  }

  @override
  Widget build(BuildContext context) {
    final displayName = component['authorName'] as String? ?? component['displayName'] as String? ?? '';
    final handle = component['authorHandle'] as String? ?? component['handle'] as String? ?? '';
    final avatarUrl = component['authorAvatarUrl'] as String?;
    final verified = component['verified'] as bool? ?? false;
    final body = component['body'] as String? ?? '';
    final likes = component['likes'] as int? ?? 0;
    final reposts = component['reposts'] as int? ?? 0;
    final replies = component['replies'] as int? ?? 0;
    final views = component['views'] as int? ?? 0;
    final mediaChild = component['mediaChild'] as String?;
    final children = mediaChild != null ? [mediaChild] : (component['children'] as List<dynamic>?)?.cast<String>() ?? [];

    return Container(
      decoration: BoxDecoration(
        color: Colors.black,
        border: Border.all(color: const Color(0xFF2F3336)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                CircleAvatar(
                  radius: 20,
                  backgroundColor: const Color(0xFF2F3336),
                  backgroundImage: avatarUrl != null && avatarUrl.isNotEmpty ? NetworkImage(avatarUrl) : null,
                  child: avatarUrl == null || avatarUrl.isEmpty
                      ? Text(displayName.isNotEmpty ? displayName[0].toUpperCase() : '?',
                          style: const TextStyle(color: Color(0xFFE7E9EA), fontWeight: FontWeight.bold))
                      : null,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Row(
                    children: [
                      Text(displayName,
                          style: const TextStyle(
                              color: Color(0xFFE7E9EA), fontWeight: FontWeight.w700, fontSize: 15, height: 1.33)),
                      if (verified) ...[
                        const SizedBox(width: 4),
                        const Icon(Icons.verified, size: 16, color: Color(0xFF1D9BF0)),
                      ],
                      const SizedBox(width: 4),
                      Text(handle.startsWith('@') ? handle : '@$handle',
                          style: const TextStyle(color: Color(0xFF71767B), fontSize: 15, height: 1.33)),
                      const Text(' · ',
                          style: TextStyle(color: Color(0xFF71767B), fontSize: 15)),
                      const Text('1h',
                          style: TextStyle(color: Color(0xFF71767B), fontSize: 15)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          // Body
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
            child: Text(body,
                style: const TextStyle(color: Color(0xFFE7E9EA), fontSize: 15, height: 1.33)),
          ),
          // Media child
          for (final childId in children)
            buildComponent(childId, components, surfaceId, state),
          // Engagement
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _engagementItem(Icons.chat_bubble_outline, _formatNumber(replies)),
                _engagementItem(Icons.repeat, _formatNumber(reposts)),
                _engagementItem(Icons.favorite_outline, _formatNumber(likes)),
                _engagementItem(Icons.bar_chart, _formatNumber(views)),
                const Icon(Icons.bookmark_border, size: 16, color: Color(0xFF71767B)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _engagementItem(IconData icon, String count) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: const Color(0xFF71767B)),
        const SizedBox(width: 4),
        Text(count, style: const TextStyle(color: Color(0xFF71767B), fontSize: 13)),
      ],
    );
  }
}
