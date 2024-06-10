﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// ChatCompletionRequest.cs
using System.Text.Json.Serialization;
using System.Collections.Generic;

namespace AutoGen.Anthropic.DTO;

public class ChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; set; }

    [JsonPropertyName("system")]
    public string? SystemMessage { get; set; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; }

    [JsonPropertyName("metadata")]
    public object? Metadata { get; set; }

    [JsonPropertyName("stop_sequences")]
    public string[]? StopSequences { get; set; }

    [JsonPropertyName("stream")]
    public bool? Stream { get; set; }

    [JsonPropertyName("temperature")]
    public decimal? Temperature { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("top_p")]
    public decimal? TopP { get; set; }

    public ChatCompletionRequest()
    {
        Messages = new List<ChatMessage>();
    }
}

public class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; }

    [JsonPropertyName("content")]
    public string Content { get; set; }

    public ChatMessage(string role, string content)
    {
        Role = role;
        Content = content;
    }
}
