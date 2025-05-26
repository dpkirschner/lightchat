Epic 2: Core Chat Functionality (Backend)
Goal: Implement the main chat streaming logic for configured providers.

    LC-006: Implement Basic Chat Engine & Ollama Chat (SSE Streaming)
        Description: Create backend/chat_engine.py. Implement chat_stream() in OllamaProvider to support Server-Sent Events (SSE). Create POST /chat endpoint in main.py that uses chat_engine.py to orchestrate the request to OllamaProvider and streams the response.
        AC:
            chat_engine.py created with a function to handle chat requests.
            OllamaProvider.chat_stream() connects to Ollama's chat/generate endpoint and yields tokens.
            POST /chat endpoint accepts {"prompt": "...", "provider_id": "...", "model_id": "..."}.
            Endpoint returns text/event-stream with events like data: {"token": "..."}.
            Pydantic models defined for /chat request and SSE data payload.

    LC-007: Enhance Chat Engine with Retry Logic
        Description: Update chat_engine.py or the provider implementations to include a simple retry mechanism (e.g., 1 retry with a short delay) for provider API calls.
        AC:
            Provider calls (e.g., to Ollama) are retried once on transient failures (e.g., network error, timeout).
            Retry logic is configurable or clearly defined (e.g., 1 retry, 2s delay).

    LC-008: Implement OpenAI Provider Adapter (Models & Streaming Chat)
        Description: Create backend/providers/openai.py with OpenAIProvider. Implement list_models() (can return a fixed list or query OpenAI API) and chat_stream() using the OpenAI API for SSE. API key will be read from ProviderConfig (requires LC-010).
        AC:
            OpenAIProvider class created.
            list_models() method implemented.
            chat_stream() method correctly streams responses from OpenAI API.
            Handles OpenAI API errors gracefully.
            (Dependent on LC-010 for API key configuration).


Epic 3: Logging & Full Configuration Management (Backend)
Goal: Implement robust logging and full configuration capabilities including hot-reloading.

    LC-009: Implement JSONL Logging Subsystem with Rotation
        Description: Develop backend/logger.py. Implement asynchronous logging to JSONL files. Logs should include timestamp, level, message, and contextual data. Implement log file rotation (configurable max files/size).
        AC:
            logger.py setup for asynchronous JSONL logging.
            Logs are written to [AppLogDir]/LightChat/logs/YYYY-MM-DDTHH-MM-SS.chatlog.jsonl.
            Log rotation (e.g., 5 files, 10MB each) is functional.
            Basic application events (e.g., startup, errors) are logged.

    LC-010: Integrate Chat Logging & Implement Full ProviderConfig
        Description: Integrate logger.py into chat_engine.py to log user prompts and assistant responses (including provider/model details). Fully define and use ProviderConfig Pydantic model in config.py (id, type, host, api_key, system_prompt). GET /providers should now dynamically list from these settings.
        AC:
            User prompts and assistant responses (including streamed tokens/final summary) are logged with context.
            ProviderConfig model fully implemented and validated.
            settings.yaml can now store a list of ProviderConfig objects.
            GET /providers returns providers based on settings.yaml.
            Ollama and OpenAI providers load and use their specific configurations (host, API key, system_prompt).

    LC-011: Implement PUT /config Endpoint & Settings Hot-Reload
        Description: Create PUT /config endpoint in main.py to allow authenticated (initially none for local child process) updates to settings.yaml. Implement hot-reloading of settings (especially provider list and their configs) using watchfiles.
        AC:
            PUT /config accepts JSON payload for AppConfig and list[ProviderConfig].
            Validates payload and overwrites settings.yaml.
            watchfiles monitors settings.yaml in the backend.
            Changes to settings.yaml (e.g., adding a provider, changing an API key) are reflected in the running application without restart (e.g., /providers shows new provider, chat uses new API key).

Epic 4: Frontend Foundation & Tauri Integration
Goal: Set up the Tauri React frontend and establish communication with the backend.

    LC-012: Initialize Tauri + React Project & Basic UI Shell
        Description: Create the Tauri project using the React + TypeScript template. Implement a very basic application shell (e.g., a main content area, placeholders for sidebars/selectors).
        AC:
            Tauri project (src-tauri, src folders) successfully created.
            Basic React components (App.tsx, simple layout) are in place.
            npm run tauri dev launches the application window showing the basic shell.

    LC-013: Implement Tauri IPC & Backend Dev Integration
        Description: Create src/tauri/api.ts providing typed wrappers for invoke calls. Configure tauri.conf.json's beforeDevCommand to launch the FastAPI backend. Implement an IPC call to the backend's /health endpoint and display its status in the UI.
        AC:
            api.ts contains at least one function (e.g., checkBackendHealth).
            FastAPI backend successfully starts as a child process when npm run tauri dev is executed.
            Frontend UI displays "Backend Healthy" or "Backend Error" based on the /health IPC call.

    LC-014: Implement Provider & Model Selector Components
        Description: Create ProviderSelector.tsx and ModelSelector.tsx. ProviderSelector fetches and displays providers from GET /providers (via api.ts). Selecting a provider populates ModelSelector by fetching from GET /models/{provider_id}. Use React Context (AppState) for selected provider/model.
        AC:
            ProviderSelector correctly lists providers.
            Changing provider in ProviderSelector updates ModelSelector with correct models.
            Selected provider and model IDs are stored in AppState.
            UI handles loading states while fetching data.

Epic 5: Frontend Chat Interface & Features
Goal: Build out the main chat UI, message streaming, and other user-facing features.

    LC-015: Implement Chat Window & Message Rendering
        Description: Create ChatWindow.tsx and ChatMessage.tsx. ChatWindow should display a list of messages from chatHistory in AppState. Style user and assistant messages distinctly.
        AC:
            ChatWindow renders messages.
            ChatMessage component styles messages based on role (user/assistant).
            Chat window auto-scrolls to the newest message.
            (Initially, chatHistory can be populated with mock data for UI development).

    LC-016: Implement Chat Input & Integrate /chat SSE Stream
        Description: Create a chat input component (PromptInput.tsx). On message submission, use api.ts to call the backend POST /chat IPC command. Handle the SSE stream events to update ChatWindow and chatHistory in AppState in real-time.
        AC:
            User can type and submit messages via PromptInput.tsx.
            Frontend makes a request to the backend's /chat functionality via IPC.
            Assistant's response tokens are streamed into ChatWindow progressively.
            chatHistory in AppState is accurately updated.
            A "thinking" or loading indicator is shown while waiting for the first token.

    LC-017: Implement Log Drawer Component
        Description: Create LogDrawer.tsx. This component, when visible, will read log files from the directory specified in AppConfig (obtained via an IPC call to get config or a dedicated API endpoint returning log path). Display the most recent N (e.g., 100) log lines.
        AC:
            LogDrawer is a toggleable UI component.
            It uses Tauri's fs API (via an IPC wrapper if needed) to read log files from the configured directory.
            Displays the last ~100 lines from the latest log file, auto-scrolling to the bottom.
            Handles cases where log files/directory might not exist.
            Log visibility state (logVisible) managed in AppState.

    LC-018: Implement Session System Prompt Editor
        Description: Create PromptEditor.tsx. This UI element allows the user to input text that will be prepended to their next chat message's prompt for the current session only. This system prompt is managed entirely in frontend state.
        AC:
            PromptEditor input field is present in the UI.
            Text entered into PromptEditor is included with the user's prompt in the data sent to the /chat IPC call.
            The PromptEditor's content can be cleared or managed as appropriate after a message is sent.

Epic 6: Testing & Quality Assurance
Goal: Ensure the application is robust and reliable through automated testing.

    LC-019: Backend Unit & Integration Tests (Part 1)
        Description: Write pytest tests for: config.py (load/save, Pydantic models), GET /health, GET /providers, GET /models/{provider_id} (using respx to mock Ollama/OpenAI HTTP calls).
        AC:
            Unit tests for configuration logic achieve >80% coverage.
            Integration tests for specified metadata endpoints pass.
            Mocked provider interactions are tested.

    LC-020: Backend Unit & Integration Tests (Part 2 - Chat & Logging)
        Description: Write pytest tests for: POST /chat SSE streaming (mocking provider streams), chat_engine.py logic including retries, logger.py (verify log format for sample messages).
        AC:
            Tests for /chat verify SSE stream format and content.
            Retry mechanism in chat_engine.py is tested.
            Sample log entries are validated against the expected JSONL schema.

    LC-021: Frontend Unit & Component Tests (Part 1 - Selectors, Display)
        Description: Write vitest and React Testing Library tests for: ProviderSelector, ModelSelector, ChatMessage components. Test basic AppState updates related to selections. Mock IPC calls from api.ts.
        AC:
            Component tests for selectors verify data rendering and basic interactions.
            ChatMessage rendering tested.
            IPC calls are mocked successfully in tests.

    LC-022: Frontend Unit & Component Tests (Part 2 - Chat Logic, Log Drawer)
        Description: Write vitest tests for: chat input submission logic, SSE event handling and chatHistory updates in AppState, LogDrawer functionality (mocking Tauri FS calls).
        AC:
            Tests verify chat submission and chatHistory state changes.
            SSE data parsing and display logic tested.
            LogDrawer component tests cover display of mock log data.

Epic 7: Packaging & CI
Goal: Prepare the application for distribution and set up continuous integration.

    LC-023: Configure Tauri macOS Build & Basic CI Pipeline
        Description: Ensure npm run tauri build successfully generates a runnable macOS .app. Set up a GitHub Actions workflow that runs linters (backend & frontend) and all automated tests (backend & frontend) on every push to main and on pull requests.
        AC:
            npm run tauri build produces a working macOS application.
            GitHub Actions workflow file created.
            Workflow executes all lint tasks (e.g., [P2.1.A] Lint All Code from tasks.json).
            Workflow executes all test tasks (e.g., [P2.2.A] Test All Code from tasks.json).
            Build status (pass/fail) is correctly reported in GitHub.

    LC-024: (Optional/Stretch) Basic E2E Test with Playwright
        Description: Set up Playwright for E2E testing. Write one simple E2E test: launch the app, send a message using the default provider/model (Ollama, potentially with a small, predictable local model), and verify a response appears in the chat window.
        AC:
            Playwright is configured in the project.
            A basic E2E test for the main chat flow runs successfully locally against the dev environment.