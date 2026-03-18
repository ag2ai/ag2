import 'package:flutter/material.dart';

import '../../state/chat_state.dart';

class A2UITextField extends StatelessWidget {
  final Map<String, dynamic> component;
  final String surfaceId;
  final ChatState state;

  const A2UITextField({
    super.key,
    required this.component,
    required this.surfaceId,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final placeholder = component['placeholder'] as String? ?? '';
    final dataBinding = component['dataBinding'] as Map<String, dynamic>?;
    final path = dataBinding?['path'] as String?;

    final initial = path != null
        ? state.dataModelService.resolveDataPath(surfaceId, path)?.toString() ?? ''
        : '';

    return TextField(
      controller: TextEditingController(text: initial),
      style: const TextStyle(color: Color(0xFFE2E8F0), fontSize: 14),
      decoration: InputDecoration(
        hintText: placeholder,
        hintStyle: const TextStyle(color: Color(0xFF94A3B8)),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      ),
      onChanged: (value) {
        if (path != null) {
          state.dataModelService.setDataPath(surfaceId, path, value);
        }
      },
    );
  }
}
