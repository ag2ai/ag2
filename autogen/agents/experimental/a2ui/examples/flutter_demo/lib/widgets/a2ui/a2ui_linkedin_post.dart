import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

/// Custom LinkedInPost component — matches real LinkedIn post formatting.
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

  String _formatNumber(int n) {
    if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}K';
    return '$n';
  }

  @override
  Widget build(BuildContext context) {
    final authorName = component['authorName'] as String? ?? '';
    final authorTitle = component['authorHeadline'] as String? ?? component['authorTitle'] as String? ?? '';
    final avatarUrl = component['authorAvatarUrl'] as String?;
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
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(color: Colors.black.withValues(alpha: 0.08), blurRadius: 0, spreadRadius: 1),
          BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 3, offset: const Offset(0, 2)),
        ],
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
                  radius: 24,
                  backgroundColor: const Color(0xFFE0E0E0),
                  backgroundImage: avatarUrl != null && avatarUrl.isNotEmpty ? NetworkImage(avatarUrl) : null,
                  child: avatarUrl == null || avatarUrl.isEmpty
                      ? Text(authorName.isNotEmpty ? authorName[0].toUpperCase() : '?',
                          style: const TextStyle(color: Color(0xFF666666), fontWeight: FontWeight.bold, fontSize: 18))
                      : null,
                ),
                const SizedBox(width: 8),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(authorName,
                        style: const TextStyle(
                            color: Color(0xE5000000), fontWeight: FontWeight.w600, fontSize: 16, height: 1.25)),
                    if (authorTitle.isNotEmpty)
                      Text(authorTitle,
                          style: const TextStyle(color: Color(0x99000000), fontSize: 12, height: 1.33)),
                    Text('1d · 🌐',
                        style: const TextStyle(color: Color(0x99000000), fontSize: 12, height: 1.33)),
                  ],
                ),
              ],
            ),
          ),
          // Body
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Text(body,
                style: const TextStyle(color: Color(0xE5000000), fontSize: 14, height: 1.43)),
          ),
          // Hashtags
          if (hashtags.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
              child: Text(
                hashtags.map((h) => h.startsWith('#') ? h : '#$h').join(' '),
                style: const TextStyle(color: Color(0xFF0A66C2), fontSize: 14, fontWeight: FontWeight.w600, height: 1.43),
              ),
            ),
          // Media child
          for (final childId in children) ...[
            const SizedBox(height: 12),
            buildComponent(childId, components, surfaceId, state),
          ],
          // Engagement counts
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
            child: Row(
              children: [
                const Text('👍 ', style: TextStyle(fontSize: 12)),
                Text(_formatNumber(likes), style: const TextStyle(color: Color(0x99000000), fontSize: 12)),
                const Text(' · ', style: TextStyle(color: Color(0x99000000), fontSize: 12)),
                Text('${_formatNumber(comments)} comments', style: const TextStyle(color: Color(0x99000000), fontSize: 12)),
                const Text(' · ', style: TextStyle(color: Color(0x99000000), fontSize: 12)),
                Text('${_formatNumber(reposts)} reposts', style: const TextStyle(color: Color(0x99000000), fontSize: 12)),
              ],
            ),
          ),
          // Action buttons
          Container(
            decoration: const BoxDecoration(
              border: Border(top: BorderSide(color: Color(0xFFE0E0E0), width: 1)),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Row(
              children: [
                _actionButton(Icons.thumb_up_outlined, 'Like'),
                _actionButton(Icons.comment_outlined, 'Comment'),
                _actionButton(Icons.repeat, 'Repost'),
                _actionButton(Icons.send_outlined, 'Send'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionButton(IconData icon, String label) {
    return Expanded(
      child: TextButton.icon(
        onPressed: null,
        icon: Icon(icon, size: 18, color: const Color(0x99000000)),
        label: Text(label,
            style: const TextStyle(color: Color(0x99000000), fontSize: 14, fontWeight: FontWeight.w600)),
      ),
    );
  }
}
