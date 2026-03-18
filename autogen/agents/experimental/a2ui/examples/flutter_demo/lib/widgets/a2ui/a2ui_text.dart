import 'package:flutter/material.dart';

class A2UIText extends StatelessWidget {
  final Map<String, dynamic> component;

  const A2UIText({super.key, required this.component});

  @override
  Widget build(BuildContext context) {
    final text = component['text'] as String? ?? '';
    final variant = component['variant'] as String? ?? 'body';

    final style = switch (variant) {
      'h1' => const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Color(0xFFE2E8F0)),
      'h2' => const TextStyle(fontSize: 20, fontWeight: FontWeight.w600, color: Color(0xFFC4B5FD)),
      'h3' => const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Color(0xFFE2E8F0)),
      'caption' => const TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
      _ => const TextStyle(fontSize: 14, color: Color(0xFFE2E8F0)),
    };

    return Text(text, style: style);
  }
}
