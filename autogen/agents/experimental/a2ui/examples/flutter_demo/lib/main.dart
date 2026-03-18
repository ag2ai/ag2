import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'services/a2a_client.dart';
import 'services/data_model_service.dart';
import 'state/chat_state.dart';
import 'widgets/chat_screen.dart';

void main() {
  runApp(const A2UIDemoApp());
}

class A2UIDemoApp extends StatelessWidget {
  const A2UIDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    final a2aClient = A2AClient(baseUrl: 'http://localhost:9000');
    final dataModelService = DataModelService();

    return ChangeNotifierProvider(
      create: (_) => ChatState(
        a2aClient: a2aClient,
        dataModelService: dataModelService,
      ),
      child: MaterialApp(
        title: 'A2UI Demo',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xFF0F172A),
          colorScheme: const ColorScheme.dark(
            primary: Color(0xFF6366F1),
            surface: Color(0xFF1E293B),
            onSurface: Color(0xFFE2E8F0),
          ),
          appBarTheme: const AppBarTheme(
            backgroundColor: Color(0xFF1E293B),
            elevation: 0,
          ),
          inputDecorationTheme: InputDecorationTheme(
            filled: true,
            fillColor: const Color(0xFF1E293B),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(10),
              borderSide: const BorderSide(color: Color(0xFF334155)),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(10),
              borderSide: const BorderSide(color: Color(0xFF334155)),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(10),
              borderSide: const BorderSide(color: Color(0xFF6366F1)),
            ),
          ),
        ),
        home: const ChatScreen(),
      ),
    );
  }
}
