import 'package:flutter/material.dart';

import '../../state/chat_state.dart';
import 'a2ui_button.dart';
import 'a2ui_card.dart';
import 'a2ui_choice_picker.dart';
import 'a2ui_column.dart';
import 'a2ui_divider.dart';
import 'a2ui_image.dart';
import 'a2ui_linkedin_post.dart';
import 'a2ui_row.dart';
import 'a2ui_text.dart';
import 'a2ui_text_field.dart';
import 'a2ui_x_post.dart';

/// Build a widget from a component ID by looking it up in the components map.
Widget buildComponent(
  String id,
  Map<String, Map<String, dynamic>> components,
  String surfaceId,
  ChatState state,
) {
  final comp = components[id];
  if (comp == null) return const SizedBox.shrink();

  final type = comp['component'] as String?;

  return switch (type) {
    'Text' => A2UIText(component: comp),
    'Button' => A2UIButton(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    'Row' => A2UIRow(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    'Column' => A2UIColumn(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    'Card' => A2UICard(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    'Image' => A2UIImage(component: comp),
    'Divider' => const A2UIDivider(),
    'TextField' => A2UITextField(
        component: comp,
        surfaceId: surfaceId,
        state: state,
      ),
    'ChoicePicker' => A2UIChoicePicker(
        component: comp,
        surfaceId: surfaceId,
        state: state,
      ),
    'LinkedInPost' => A2UILinkedInPost(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    'XPost' => A2UIXPost(
        component: comp,
        components: components,
        surfaceId: surfaceId,
        state: state,
      ),
    _ => const SizedBox.shrink(),
  };
}
