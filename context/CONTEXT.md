# LightChat v0.1: System Architecture

**Document Purpose**: This document outlines the system architecture for LightChat version 0.1. It details the major components, their interactions, key technologies, data flows, and design principles. This information is intended for developers working on the project and for language models needing to understand the project's technical vision.

## 1. Overview

* **Application Name**: LightChat
* **Version**: 0.1 (Minimum Viable Product - MVP)
* **Purpose**: LightChat is a desktop AI chat application designed to provide a unified, responsive interface to various Large Language Model (LLM) providers. It emphasizes support for local models (e.g., via Ollama) and offers user-configurable settings.
* **Core Philosophy**:
    * **Local-First**: Prioritize robust support for local LLM providers.
    * **Responsive Interface**: Ensure a fast, streaming user experience for chat interactions.
    * **User Control**: Provide clear configuration options through accessible files.
    * **Observability**: Implement comprehensive logging for debugging and transparency.

## 2. High-Level Architecture

* **Application Type**: Native Desktop Application.
* **Initial Target Platform (v0.1)**: macOS.
* **Process Structure**: LightChat operates with two primary processes:
    1.  **Frontend Process**: A Tauri-managed webview executing a React (TypeScript) Single-Page Application (SPA). This process handles all user interface elements and user interactions.
    2.  **Backend Process**: A Python-based FastAPI server. This process is managed as a child process by Tauri and handles core application logic, including communication with LLM providers, configuration management, and logging.
* **Interaction Model**:
    * The Frontend communicates with the Backend primarily through Tauri's Inter-Process Communication (IPC) mechanism.
    * Tauri IPC calls from the Frontend are typically mapped to HTTP requests targeting the local FastAPI server running as the Backend process.

## 3. Key Components

### 3.1. Frontend (Tauri + React)

* **Desktop Shell**: Tauri (v1.x).
    * Manages the webview environment.
    * Handles native system interactions (e.g., file system access for paths, window management).
    * Manages the lifecycle of the Backend child process.
* **UI Framework**: React (v18.x) with TypeScript.
* **State Management (v0.1)**:
    * Utilizes React Context API (`AppState`) for global state.
    * **For v0.1, frontend state is ephemeral**: `chatHistory`, selected provider/model, and other UI states (e.g., `logVisible`) reset when the application is closed.
* **IPC Layer**:
    * Located in `src/tauri/api.ts`.
    * Provides strongly-typed wrapper functions around Tauri's `invoke` API to communicate with the Backend.
* **Key UI Components**:
    * `ProviderSelector`: Dropdown to select the active LLM provider.
    * `ModelSelector`: Dropdown to select a model from the active provider.
    * `ChatWindow`: Displays the conversation history and streams new messages.
    * `LogDrawer`: A toggleable view displaying formatted JSONL logs.
    * `PromptEditor`: An input field for a session-specific system prompt that prepends to the user's next message. Not persisted.

### 3.2. Backend Service (FastAPI)

* **Framework**: FastAPI (Python 3.9+).
* **Entry Point**: `backend/main.py`. Can be run directly using `python -m backend`.
* **Server**: Uvicorn, listening on `127.0.0.1:8000` by default. Access is restricted to the local machine.
* **Tauri Integration**:
    * The Backend FastAPI server is started as a child process by Tauri before the dev server (`beforeDevCommand`) and before the build process (`beforeBuildCommand`) as defined in `tauri.conf.json`.
* **API Surface**:
    * `POST /chat`: Accepts user prompt and streams assistant's response tokens via Server-Sent Events (`text/event-stream`).
    * `GET /providers`: Returns a list of configured LLM providers with their metadata (id, name, type, status).
    * `GET /models/{provider_id}`: Returns a list of available models for the specified provider.
    * `PUT /config`: Accepts updated settings data, validates, and persists it to the configuration file.
    * `GET /health`: A simple health check endpoint returning `{"status": "ok"}`.
* **Error Handling**: API errors return a structured JSON response:
    ```json
    {
      "error_code": "UNIQUE_ERROR_CODE",
      "message": "Descriptive error message.",
      "details": { /* Optional contextual information */ }
    }
    ```
* **Core Modules**:
    * `chat_engine.py`: Orchestrates the chat logic: receives prompt, interacts with the appropriate LLM provider, handles streaming, and coordinates logging. Implements a simple retry mechanism (1 retry) for provider requests.
    * `providers/`: Contains adapters for different LLM providers.
        * `base.py`: Defines an abstract base class (`LLMProvider`) with methods like `list_models()` and `chat_stream()`.
        * `ollama.py`: Implements the `LLMProvider` interface for Ollama (local HTTP communication).
        * `openai.py`: Implements the `LLMProvider` interface for OpenAI API.
    * `config.py`: Handles loading and saving of application settings from/to `settings.yaml` using Pydantic models for validation and serialization.
    * `logger.py`: Manages the logging subsystem, including formatting and file rotation. Provides hooks for pre-send and post-receive logging in the chat flow.

### 3.3. Configuration System

* **Format (v0.1)**: YAML. The primary configuration file is `settings.yaml`.
* **Location**: Stored in the OS-specific application configuration directory (e.g., `[AppConfigDir]/LightChat/settings.yaml`). Path resolved using Tauri's path API.
* **Schema Definition**: Pydantic models define the structure:
    * `ProviderConfig`: Includes `id` (str), `type` (enum: "ollama", "openai"), `api_key` (str, optional), `host` (str, optional), `system_prompt` (str, optional).
    * `AppConfig`: Includes `default_provider` (str, optional), `log_dir` (str, optional), `logging_enabled` (bool).
* **Management**: Loaded by the Backend on startup and saved via the `PUT /config` endpoint.
* **Hot Reload**: The Backend uses the `watchfiles` library to monitor `settings.yaml` for changes. If changes are detected, relevant parts of the configuration (like provider instances) are reloaded without requiring an application restart.

### 3.4. Logging Subsystem

* **Format**: JSON Lines (JSONL). Each log entry is a self-contained JSON object on a new line.
* **Location**: Stored in the OS-specific application log directory (e.g., `[AppLogDir]/LightChat/logs/`). Filenames are timestamped (e.g., `2025-05-26T10-30-00.chatlog.jsonl`). Path resolved using Tauri's path API.
* **Schema Example**:
    ```json
    {
      "ts": "2025-05-26T10:30:00.123Z", // ISO 8601 Timestamp
      "role": "user" | "assistant" | "system" | "error", // Source of the message or log level
      "content": "Log message or chat content.", // For chat, the message string
      "provider_id": "ollama_local", // If applicable
      "model_id": "llama3.2", // If applicable
      // Additional fields for errors or specific system events
      "event_type": "config_loaded", // For system events
      "details": { /* Context-specific details for errors/events */ }
    }
    ```
* **Features**:
    * **Asynchronous Logging**: Operations are designed to be non-blocking to minimize performance impact on the chat stream.
    * **Rotation**: Configurable maximum number of log files (e.g., 5) and maximum size per file (e.g., 10MB). Oldest files are deleted when limits are exceeded.
    * **Toggleable**: Logging can be enabled/disabled per application run (session), defaulting to on.

## 4. Data Flows

### 4.1. Chat Request and Streaming Response

1.  **User Input (Frontend)**: User types a message in `ChatWindow` and submits. The session-specific `PromptEditor` content is prepended if present.
2.  **IPC Call (Frontend)**: `src/tauri/api.ts` invokes a Tauri command (e.g., `chat_stream`) with the prompt data.
3.  **HTTP Request (Tauri -> Backend)**: Tauri relays this as an HTTP `POST` request to the Backend's `/chat` endpoint.
4.  **Processing (Backend - `chat_engine.py`)**:
    a.  Logs the incoming user prompt (via `logger.py`).
    b.  Identifies the active LLM provider and model from the current configuration.
    c.  Calls the `chat_stream()` method on the corresponding provider adapter in `providers/`.
    d.  The provider adapter makes an HTTP request to the actual LLM service (Ollama, OpenAI API).
    e.  As tokens are received from the LLM, they are immediately sent back to the client via Server-Sent Events (SSE) using FastAPI's `EventSourceResponse`. Each event typically looks like `data: {"token": "Hello"}`.
    f.  Assistant response tokens are logged as they are processed (via `logger.py`).
    g.  If a provider request fails, a single retry is attempted before returning an error.
5.  **Streaming to UI (Backend -> Tauri -> Frontend)**: SSE events are streamed back through Tauri to the React application.
6.  **UI Update (Frontend)**: `ChatWindow` receives the token events and progressively renders the assistant's response.

### 4.2. Configuration Update

1.  **User Action (Frontend)**: User modifies a setting (e.g., changes the default provider through a settings interface).
2.  **IPC Call (Frontend)**: `src/tauri/api.ts` invokes a Tauri command (e.g., `update_config`) with the new settings data.
3.  **HTTP Request (Tauri -> Backend)**: Tauri relays this as an HTTP `PUT` request to the Backend's `/config` endpoint.
4.  **Processing (Backend - `config.py`)**:
    a.  The incoming data is validated against Pydantic models.
    b.  If valid, the `settings.yaml` file is updated.
    c.  A success or error response is returned to the Frontend.
5.  **Hot Reload (Backend)**: Independently, `watchfiles` detects the modification to `settings.yaml` and triggers a reload of the relevant application configurations (e.g., re-initializes provider clients with new API keys or endpoints).

## 5. Key Technologies & Libraries

* **Backend**: Python (3.9+), FastAPI, Pydantic, Uvicorn, `watchfiles`, `httpx` (for HTTP clients to LLM providers).
* **Frontend**: TypeScript, React (v18+), Tauri (v1+).
* **Testing**:
    * Backend: `pytest`, `pytest-asyncio`, `respx` (for mocking HTTP requests).
    * Frontend: `vitest`, React Testing Library.
    * End-to-End (E2E): Playwright.
* **Build & Packaging**: Tauri CLI.
* **Continuous Integration (CI)**: GitHub Actions (for testing, building, notarizing).

## 6. Design Principles & Decisions

* **Modularity**: Clear separation of concerns between Frontend, Backend, and individual modules within the Backend (e.g., `chat_engine`, `providers`, `config`, `logger`). Provider adapters are designed to be pluggable.
* **Local-First Emphasis**: Strong initial support and prioritization for local LLM providers like Ollama.
* **Responsiveness**: Achieved through SSE for chat message streaming and asynchronous operations in the Backend.
* **User Control & Transparency**: Configuration is managed via a human-readable YAML file. Logging provides insight into application behavior.
* **Developer Experience**: Leveraging modern frameworks and tools. Strong typing with Python type hints and TypeScript enhances maintainability.
* **Simplicity for v0.1 (MVP Focus)**:
    * Frontend state (chat history, UI settings) is not persisted across sessions.
    * Configuration is handled exclusively through YAML.
    * Core functionality (chat with configurable providers) is prioritized over extensive features.

## 7. Packaging & Distribution (v0.1)

* **Build Command**: `npm run tauri build`.
* **Output (macOS)**: A signed `.app` bundle.
* **Signing & Notarization**: Utilizes Apple Developer ID for signing. Hardened runtime is enabled. Notarization via GitHub Actions.
* **Application Updates**: Tauri's built-in auto-updater mechanism, configured to check GitHub Releases for new versions.

## 8. Future Epics (Post-MVP Considerations)

The v0.1 architecture is intended to be a foundation for future enhancements. Key areas for future development include:

* **Tool/Function Calling Integration**: Extending the provider interface and `chat_engine` to support LLM tool use.
* **Agent Orchestrator**: Evolving `chat_engine.py` into a more sophisticated agent runtime with planning and memory capabilities.
* **Multi-Session UI**: Implementing features like tabbed Browse for multiple concurrent chat sessions.
* **Configuration File Format**: Adding support for TOML (`settings.toml`) as an alternative configuration format.
* **Frontend State Persistence**: Implementing persistence for `chatHistory`, active provider/model selections, and other UI states across application restarts.
* **Expanded LLM Provider Support**: Integrating more third-party and local LLM provider adapters.
* **Cross-Platform Support**: Officially building and distributing for Windows and Linux.