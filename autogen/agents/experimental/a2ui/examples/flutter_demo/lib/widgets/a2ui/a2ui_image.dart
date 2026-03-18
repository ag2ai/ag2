import 'package:flutter/material.dart';

class A2UIImage extends StatelessWidget {
  final Map<String, dynamic> component;

  const A2UIImage({super.key, required this.component});

  @override
  Widget build(BuildContext context) {
    final src = component['url'] as String? ?? component['src'] as String? ?? '';
    final alt = component['alt'] as String? ?? '';

    if (src.isEmpty) return const SizedBox.shrink();

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.network(
        src,
        fit: BoxFit.cover,
        errorBuilder: (_, __, ___) => Container(
          height: 120,
          decoration: BoxDecoration(
            color: const Color(0xFF334155),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(
            child: Text(
              alt.isNotEmpty ? alt : 'Image',
              style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12),
            ),
          ),
        ),
      ),
    );
  }
}
