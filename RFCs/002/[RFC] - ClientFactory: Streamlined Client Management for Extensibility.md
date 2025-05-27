# [RFC] - ClientFactory: Streamlined Client Management for Extensibility

**Author:** Davor Runje, Tvrtko Sternak, Davorin Ruševljan
**Status:** Discussion

## **Introduction**

The current implementation of client creation and management has enabled the development of various clients with different capabilities. However, the existing approach has significant limitations, particularly in terms of maintainability and extensibility. This RFC proposes a `ClientFactory` model to address these issues by introducing a structured, extensible, and scalable framework for managing client creation.

## **Pain Points in the Current Model**

1. **Bloated Client Implementations:**
   - Client `create` methods are filled with conditional logic (`if`/`else`) to handle multiple cases.
   - This violates the **Single Responsibility Principle** by mixing logic for different functionalities.

2. **Hard-Coded Matching Logic:**
   - The client wrapper manually determines which client to use based on configuration, making it difficult to introduce new clients.

3. **Extensibility Challenges:**
   - Adding new clients or features requires modifying existing logic, increasing the risk of introducing errors.

4. **Inconsistent Prioritization:**
   - The system does not have a clear mechanism to handle cases where multiple clients match a configuration, leading to unpredictable behavior.

## **What We Would Like to Achieve (Goals)**

A streamlined client creation model that:
1. **Encourages Modular Design:**
   - Each client should focus solely on its unique functionality, adhering to the **Single Responsibility Principle**.

2. **Supports Extensibility:**
   - Adding new clients should be as simple as registering them with the factory.

3. **Handles Specificity Gracefully:**
   - When multiple clients match a configuration, the most specific client should be selected automatically.

4. **Backward Compatibility:**
   - The model should support existing clients with minimal or no changes.

## **Proposed Solution: ClientFactory**

The `ClientFactory` provides a central mechanism for registering and creating clients. Each client specifies:
- A method to determine if it matches a given configuration (`accept`).
- A priority score to resolve conflicts when multiple clients match.

### **Core Components**

#### **Client Protocol**
A standard interface for all clients to ensure consistency:
- `create`: Generates responses based on conversation history.
- `accept`: Determines if the client can handle a given configuration and assigns a specificity score.

```python
from typing import Any, Protocol, runtime_checkable

Message = dict[str, Any]
Configuration = dict[str, Any]
ClientMessage = dict[str, Any]

@runtime_checkable
class ClientProtocol(Protocol):
    def create(self, history: Message) -> ClientMessage:
        """Generates a response based on the conversation history."""
        pass

    @classmethod
    def accept(cls, config: Configuration) -> tuple[bool, int]:
        """
        Returns:
        - A boolean indicating if the client matches.
        - An integer representing the specificity (higher is more specific).
        """
        pass
```

#### **ClientFactory**
A factory class to manage client registration and creation:
- Maintains a registry of available clients.
- Resolves conflicts based on priority when multiple clients match.

```python
class ClientFactory:
    _registry: list[ClientProtocol] = []

    @classmethod
    def register(cls, client_class: ClientProtocol) -> ClientProtocol:
        """Registers a new client class."""
        cls._registry.append(client_class)
        return client_class

    @classmethod
    def create_client(cls, config: Configuration) -> ClientProtocol:
        """Creates a client instance based on the provided configuration."""
        matches = [
            (priority, client_class)
            for client_class in cls._registry
            if (is_match := client_class.accept(config))[0]
        ]

        if not matches:
            raise ValueError(f"No compatible client found for configuration: {config}")

        # Sort by priority (highest priority first)
        matches.sort(key=lambda x: x[0], reverse=True)

        # Log a warning if multiple matches are found
        if len(matches) > 1:
            print(f"Warning: Multiple clients matched. Using {matches[0][1].__name__} with priority {matches[0][0]}.")

        return matches[0][1](config)
```

#### **Example Clients**
Two clients demonstrate the model's extensibility and specificity:
1. **DefaultOpenAIClient** (general-purpose).
2. **OpenAIStructuredClient** (more specific, supports structured output).

```python
@ClientFactory.register
class DefaultOpenAIClient:
    def __init__(self, config: Configuration):
        self.name = "DefaultOpenAIClient"

    def create(self, history: Message) -> ClientMessage:
        return "This is a response from DefaultOpenAIClient."

    @classmethod
    def accept(cls, config: Configuration) -> tuple[bool, int]:
        is_match = config.get("api_type") == "openai"
        priority = 1  # Lower specificity
        return is_match, priority


@ClientFactory.register
class OpenAIStructuredClient:
    def __init__(self, config: Configuration):
        self.name = "OpenAIStructuredClient"

    def create(self, history: Message) -> ClientMessage:
        return """{"response": "This is a response from OpenAIStructuredClient."}"""

    @classmethod
    def accept(cls, config: Configuration) -> tuple[bool, int]:
        is_match = config.get("api_type") == "openai" and "structured_output" in config
        priority = 10  # Higher specificity
        return is_match, priority
```

## **Benefits**

1. **Separation of Concerns:**
   - Clients focus only on their unique responsibilities.

2. **Ease of Extensibility:**
   - New clients can be added by simply implementing the protocol and registering with the factory.

3. **Predictable Behavior:**
   - Conflicts are resolved using priority scores.

4. **Backward Compatibility:**
   - Legacy clients can be supported with a default priority of `0`.


## **Future Considerations**

1. **Additional Prioritization Rules:**
   - Introduce finer-grained control over conflict resolution if needed.

2. **Dynamic Configuration Validation:**
   - Ensure configurations are validated before being passed to clients.

3. **Plugin System:**
   - Allow third-party developers to contribute custom clients.

## **Example Usage**

```python
# General configuration
config_general = {"api_type": "openai"}

# Structured configuration
config_structured = {"api_type": "openai", "structured_output": True}

# Create clients
general_client = ClientFactory.create_client(config_general)
print(f"Client: {general_client.name}, Response: {general_client.create([])}")

structured_client = ClientFactory.create_client(config_structured)
print(f"Client: {structured_client.name}, Response: {structured_client.create([])}")
```

## **Conclusion**

The proposed `ClientFactory` model addresses the pain points in the current implementation by promoting modularity, extensibility, and clarity. It lays a strong foundation for future enhancements while maintaining backward compatibility.
