import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

class A2UICard extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UICard({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final childId = component['child'] as String?;
    final children = childId != null ? [childId] : (component['children'] as List<dynamic>?)?.cast<String>() ?? [];

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
          for (final childId in children)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: buildComponent(childId, components, surfaceId, state),
            ),
        ],
      ),
    );
  }
}
