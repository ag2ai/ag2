import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'component_factory.dart';

class A2UIColumn extends StatelessWidget {
  final Map<String, dynamic> component;
  final Map<String, Map<String, dynamic>> components;
  final String surfaceId;
  final ChatState state;

  const A2UIColumn({
    super.key,
    required this.component,
    required this.components,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final children = (component['children'] as List<dynamic>?)?.cast<String>() ?? [];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        for (final childId in children)
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: buildComponent(childId, components, surfaceId, state),
          ),
      ],
    );
  }
}
