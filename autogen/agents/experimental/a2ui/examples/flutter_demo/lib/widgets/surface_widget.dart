import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../state/chat_state.dart';
import 'a2ui/component_factory.dart';

/// Renders an A2UI surface by processing its operations and building a component tree.
class SurfaceWidget extends StatelessWidget {
  final String surfaceId;

  const SurfaceWidget({super.key, required this.surfaceId});

  @override
  Widget build(BuildContext context) {
    return Consumer<ChatState>(
      builder: (context, state, _) {
        final ops = state.surfaces[surfaceId];
        if (ops == null || ops.isEmpty) return const SizedBox.shrink();

        // Build component map from operations
        final components = <String, Map<String, dynamic>>{};
        String? rootId;

        for (final op in ops) {
          if (op.containsKey('createSurface')) {
            // createSurface doesn't define rootId; we find root from components
          } else if (op.containsKey('updateComponents')) {
            final payload = op['updateComponents'] as Map<String, dynamic>;
            final comps = payload['components'] as List<dynamic>? ?? [];
            for (final c in comps) {
              final comp = Map<String, dynamic>.from(c as Map);
              final id = comp['id'] as String?;
              if (id != null) components[id] = comp;
            }
          }
        }

        // Find root: use 'root' id if present, otherwise first component
        rootId = components.containsKey('root') ? 'root' : (components.keys.isNotEmpty ? components.keys.first : null);

        if (rootId == null || components.isEmpty) return const SizedBox.shrink();

        return Container(
          margin: const EdgeInsets.symmetric(vertical: 8),
          child: buildComponent(
            rootId,
            components,
            surfaceId,
            state,
          ),
        );
      },
    );
  }
}
