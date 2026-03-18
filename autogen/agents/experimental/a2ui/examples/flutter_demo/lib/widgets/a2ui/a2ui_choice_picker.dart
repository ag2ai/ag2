import 'package:flutter/material.dart';

import '../../state/chat_state.dart';

class A2UIChoicePicker extends StatefulWidget {
  final Map<String, dynamic> component;
  final String surfaceId;
  final ChatState state;

  const A2UIChoicePicker({
    super.key,
    required this.component,
    required this.surfaceId,
    required this.state,
  });

  @override
  State<A2UIChoicePicker> createState() => _A2UIChoicePickerState();
}

class _A2UIChoicePickerState extends State<A2UIChoicePicker> {
  String? _selected;

  @override
  void initState() {
    super.initState();
    final valueBinding = widget.component['value'] as Map<String, dynamic>? ?? widget.component['dataBinding'] as Map<String, dynamic>?;
    final path = valueBinding?['path'] as String?;
    if (path != null) {
      _selected = widget.state.dataModelService
          .resolveDataPath(widget.surfaceId, path)
          ?.toString();
    }
  }

  @override
  Widget build(BuildContext context) {
    final options = (widget.component['options'] as List<dynamic>?)?.cast<Map<String, dynamic>>() ?? [];
    final valueBinding = widget.component['value'] as Map<String, dynamic>? ?? widget.component['dataBinding'] as Map<String, dynamic>?;
    final path = valueBinding?['path'] as String?;

    return DropdownButton<String>(
      value: _selected,
      dropdownColor: const Color(0xFF1E293B),
      style: const TextStyle(color: Color(0xFFE2E8F0), fontSize: 14),
      hint: const Text('Select...', style: TextStyle(color: Color(0xFF94A3B8))),
      isExpanded: true,
      items: options.map((opt) {
        final value = opt['value']?.toString() ?? '';
        final label = opt['label']?.toString() ?? value;
        return DropdownMenuItem(value: value, child: Text(label));
      }).toList(),
      onChanged: (value) {
        setState(() => _selected = value);
        if (path != null && value != null) {
          widget.state.dataModelService.setDataPath(widget.surfaceId, path, value);
        }
      },
    );
  }
}
