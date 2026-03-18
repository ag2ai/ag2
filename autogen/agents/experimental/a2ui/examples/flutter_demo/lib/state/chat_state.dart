import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';

import '../models/chat_message.dart';
import '../services/a2a_client.dart';
import '../services/data_model_service.dart';

const _uuid = Uuid();

class ChatState extends ChangeNotifier {
  final A2AClient a2aClient;
  final DataModelService dataModelService;
  final String contextId = _uuid.v4();

  final List<ChatMessage> messages = [];
  final Map<String, List<Map<String, dynamic>>> surfaces = {};
  bool isLoading = false;

  ChatState({required this.a2aClient, required this.dataModelService});

  /// Send a text message from the user.
  Future<void> sendMessage(String text) async {
    messages.add(UserMessage(text));
    isLoading = true;
    notifyListeners();

    try {
      final response = await a2aClient.sendMessage(text, contextId);
      _handleResponse(response);
    } catch (e) {
      messages.add(BotMessage('Error: $e'));
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  /// Send an A2UI action (button tap, etc.)
  Future<void> sendAction(Map<String, dynamic> actionPayload) async {
    isLoading = true;
    notifyListeners();

    try {
      final response = await a2aClient.sendMessage(
        '',
        contextId,
        a2uiAction: actionPayload,
      );
      _handleResponse(response);
    } catch (e) {
      messages.add(BotMessage('Error: $e'));
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  void _handleResponse(A2AResponse response) {
    // Process A2UI operations
    if (response.a2uiOperations != null) {
      _processA2UIOperations(response.a2uiOperations!);
    }

    // Add text message
    if (response.text != null && response.text!.isNotEmpty) {
      messages.add(BotMessage(response.text!));
    }
  }

  void _processA2UIOperations(List<Map<String, dynamic>> operations) {
    for (final op in operations) {
      // A2UI v0.9 messages use message-level keys (createSurface, updateComponents, etc.)
      if (op.containsKey('createSurface')) {
        final payload = op['createSurface'] as Map<String, dynamic>;
        final surfaceId = payload['surfaceId'] as String;
        final isNew = !surfaces.containsKey(surfaceId);
        surfaces[surfaceId] = [op];
        if (isNew) {
          messages.add(SurfaceMessage(surfaceId));
        }
      } else if (op.containsKey('updateComponents')) {
        final payload = op['updateComponents'] as Map<String, dynamic>;
        final surfaceId = payload['surfaceId'] as String;
        if (surfaces.containsKey(surfaceId)) {
          surfaces[surfaceId]!.add(op);
        }
      } else if (op.containsKey('updateDataModel')) {
        final payload = op['updateDataModel'] as Map<String, dynamic>;
        final surfaceId = payload['surfaceId'] as String;
        final contents = payload['dataModel'] as Map<String, dynamic>? ?? payload['contents'] as Map<String, dynamic>?;
        if (contents != null) {
          dataModelService.initModel(surfaceId, contents);
        }
        if (surfaces.containsKey(surfaceId)) {
          surfaces[surfaceId]!.add(op);
        }
      } else {
        // Unknown op — try to find surfaceId in any nested payload
        for (final value in op.values) {
          if (value is Map<String, dynamic> && value.containsKey('surfaceId')) {
            final surfaceId = value['surfaceId'] as String;
            if (surfaces.containsKey(surfaceId)) {
              surfaces[surfaceId]!.add(op);
            }
            break;
          }
        }
      }
    }
  }
}
