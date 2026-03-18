import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';

const _a2uiMimeType = 'application/json+a2ui';
const _a2uiExtensionUri = 'https://a2ui.org/a2a-extension/a2ui/v0.9';
const _uuid = Uuid();

/// Result from an A2A message/send call.
class A2AResponse {
  final String? text;
  final List<Map<String, dynamic>>? a2uiOperations;

  const A2AResponse({this.text, this.a2uiOperations});
}

/// Minimal A2A JSON-RPC client that talks directly to the backend.
class A2AClient {
  final String baseUrl;

  A2AClient({required this.baseUrl});

  /// Send a user message (with optional A2UI action DataPart) and parse the response.
  Future<A2AResponse> sendMessage(
    String text,
    String contextId, {
    Map<String, dynamic>? a2uiAction,
  }) async {
    // Build parts: TextPart + optional A2UI DataPart
    final parts = <Map<String, dynamic>>[
      {
        'kind': 'text',
        'text': text,
      },
    ];

    if (a2uiAction != null) {
      parts.add({
        'kind': 'data',
        'data': {'messages': [a2uiAction]},
        'metadata': {'mimeType': _a2uiMimeType},
      });
    }

    final body = {
      'jsonrpc': '2.0',
      'id': _uuid.v4(),
      'method': 'message/send',
      'params': {
        'message': {
          'role': 'user',
          'messageId': _uuid.v4(),
          'parts': parts,
          'contextId': contextId,
        },
        'configuration': {
          'acceptedOutputModes': ['text'],
        },
      },
    };

    final response = await http.post(
      Uri.parse(baseUrl),
      headers: {
        'Content-Type': 'application/json',
        'X-A2A-Extensions': _a2uiExtensionUri,
      },
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('A2A request failed: ${response.statusCode} ${response.body}');
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;

    if (json.containsKey('error')) {
      throw Exception('A2A error: ${json['error']}');
    }

    return _parseResult(json['result']);
  }

  /// Parse the JSON-RPC result to extract text and A2UI operations.
  A2AResponse _parseResult(Map<String, dynamic> result) {
    final artifacts = result['artifacts'] as List<dynamic>? ?? [];
    final textParts = <String>[];
    final a2uiOps = <Map<String, dynamic>>[];

    for (final artifact in artifacts) {
      final parts = (artifact['parts'] as List<dynamic>?) ?? [];
      for (final part in parts) {
        final p = part as Map<String, dynamic>;
        final kind = p['kind'] ?? p['type'];
        if (kind == 'data') {
          final metadata = p['metadata'] as Map<String, dynamic>?;
          if (metadata != null && metadata['mimeType'] == _a2uiMimeType) {
            final data = p['data'] as Map<String, dynamic>;
            final messages = data['messages'] as List<dynamic>? ?? [];
            for (final msg in messages) {
              a2uiOps.add(Map<String, dynamic>.from(msg as Map));
            }
          }
        } else if (kind == 'text') {
          final t = p['text'] as String?;
          if (t != null && t.isNotEmpty) textParts.add(t);
        }
      }
    }

    // Also check status message text
    final status = result['status'] as Map<String, dynamic>?;
    if (status != null) {
      final msg = status['message'] as Map<String, dynamic>?;
      if (msg != null) {
        final msgParts = msg['parts'] as List<dynamic>? ?? [];
        for (final part in msgParts) {
          final p = part as Map<String, dynamic>;
          final sk = p['kind'] ?? p['type'];
          if (sk == 'text') {
            final t = p['text'] as String?;
            if (t != null && t.isNotEmpty) textParts.add(t);
          }
        }
      }
    }

    return A2AResponse(
      text: textParts.isNotEmpty ? textParts.join('\n') : null,
      a2uiOperations: a2uiOps.isNotEmpty ? a2uiOps : null,
    );
  }
}
