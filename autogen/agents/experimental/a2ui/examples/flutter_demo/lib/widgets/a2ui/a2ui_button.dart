import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

class A2UIButton extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UIButton({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final action = component['action'] as Map<String, dynamic>?;
    final childId = component['child'] as String?;
    final children = (component['children'] as List<dynamic>?)?.cast<String>() ?? [];
    final label = component['text'] as String? ?? '';

    Widget child;
    if (childId != null) {
      child = buildComponent(childId, components, surfaceId, state);
    } else if (children.isNotEmpty) {
      child = buildComponent(children.first, components, surfaceId, state);
    } else {
      child = Text(label);
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF6366F1),
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        ),
        onPressed: action != null ? () => _handleAction(action) : null,
        child: child,
      ),
    );
  }

  void _handleAction(Map<String, dynamic> action) {
    final event = action['event'] as Map<String, dynamic>? ?? {};
    final name = event['name'] as String? ?? '';
    final actionContext = event['context'] as Map<String, dynamic>? ?? {};

    // Resolve any data path references in the context
    final resolvedContext = <String, dynamic>{};
    for (final entry in actionContext.entries) {
      final value = entry.value;
      if (value is Map<String, dynamic> && value.containsKey('path')) {
        final path = value['path'] as String;
        resolvedContext[entry.key] = state.dataModelService.resolveDataPath(surfaceId, path);
      } else {
        resolvedContext[entry.key] = value;
      }
    }

    final payload = {
      'version': 'v0.9',
      'action': {
        'name': name,
        'surfaceId': surfaceId,
        'context': resolvedContext,
      },
    };

    state.sendAction(payload);
  }
}
