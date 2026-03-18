/// Manages per-surface data models and resolves JSON pointer paths.
class DataModelService {
  final Map<String, Map<String, dynamic>> _models = {};

  /// Initialize or replace the data model for a surface.
  void initModel(String surfaceId, Map<String, dynamic> data) {
    _models[surfaceId] = Map<String, dynamic>.from(data);
  }

  /// Get the entire data model for a surface.
  Map<String, dynamic>? getModel(String surfaceId) => _models[surfaceId];

  /// Resolve a JSON pointer path like "/foo/bar" to its value.
  dynamic resolveDataPath(String surfaceId, String path) {
    final model = _models[surfaceId];
    if (model == null) return null;

    final segments = path.replaceAll(RegExp(r'^/'), '').split('/');
    dynamic current = model;
    for (final seg in segments) {
      if (seg.isEmpty) continue;
      if (current is Map<String, dynamic>) {
        current = current[seg];
      } else if (current is List) {
        final idx = int.tryParse(seg);
        if (idx != null && idx < current.length) {
          current = current[idx];
        } else {
          return null;
        }
      } else {
        return null;
      }
    }
    return current;
  }

  /// Set a value at a JSON pointer path.
  void setDataPath(String surfaceId, String path, dynamic value) {
    _models.putIfAbsent(surfaceId, () => {});
    final model = _models[surfaceId]!;
    final segments = path.replaceAll(RegExp(r'^/'), '').split('/');

    if (segments.length == 1) {
      model[segments[0]] = value;
      return;
    }

    dynamic current = model;
    for (var i = 0; i < segments.length - 1; i++) {
      final seg = segments[i];
      if (current is Map<String, dynamic>) {
        current.putIfAbsent(seg, () => <String, dynamic>{});
        current = current[seg];
      }
    }
    if (current is Map<String, dynamic>) {
      current[segments.last] = value;
    }
  }
}
