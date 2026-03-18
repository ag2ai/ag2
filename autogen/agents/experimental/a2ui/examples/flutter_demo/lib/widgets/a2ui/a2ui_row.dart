import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

class A2UIRow extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UIRow({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final children = (component['children'] as List<dynamic>?)?.cast<String>() ?? [];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final childId in children)
          buildComponent(childId, components, surfaceId, state),
      ],
    );
  }
}
