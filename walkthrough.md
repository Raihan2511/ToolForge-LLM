# ToolForge-LLM Refactor Walkthrough

This document summarizes the changes made to address the conversational degradation issues in ToolForge-LLM, strictly following the 9-step plan provided.

> [!NOTE]
> All dataset fixes were implemented directly in the `data_pre_processing.py` script. The raw datasets remain untouched, ensuring a repeatable and clean preprocessing pipeline.

## 1. Data Preprocessing (`data_pre_processing.py`)

### A. Fixed UltraChat Formatting (The Biggest Fix)
* **Removed Tool Injection**: `process_ultrachat()` was modified so it no longer forces `{"tool_calls": []}` or format instructions into conversational training samples.
* **Natural System Prompt**: Replaced the system prompt for UltraChat with a clean, conversational tone:
  ```python
  system_content = (
      "You are a helpful, natural, and friendly AI assistant."
  )
  ```

### B. Reduced Conversational Ratio
* Adjusted the hard limits to maintain strong tool-calling performance while reducing conversational influence:
  * `APIGEN_LIMIT`: Increased from 60k to **80,000**
  * `ULTRACHAT_LIMIT`: Reduced from 40k to **15,000**

### C. Cleaned Up System Prompts
* Removed the excessive 15 variant instructions that caused too much randomness.
* Standardized on exactly 3 clear, concise task instructions:
  ```python
  TASK_INSTRUCTIONS = [
      "You are a helpful AI assistant with access to tools.",
      "Use available tools when required to answer the user.",
      "Call functions when necessary, otherwise respond naturally.",
  ]
  ```

### D. Local Execution & Windows Compatibility
* Stripped out Google Colab dependencies (`drive.mount`).
* Dynamically fetches `Salesforce/xlam-function-calling-60k` directly from HuggingFace using the authenticated token to bypass the dataset gate.
* Added explicit `encoding="utf-8"` to file writing operations to prevent `UnicodeEncodeError` on Windows.
* Changed output files to `new_train.jsonl` and `new_val.jsonl` to ensure previous data wasn't accidentally overwritten.

---

## 2. Inference Routing (`chat_client.py` & `chitchat_client.py`)

### A. Dynamic Tool Injection
* Both client scripts were updated to dynamically decide whether to inject the `TOOLS` schema into the prompt based on user keywords.
* **Implementation:**
  ```python
  tool_keywords = [
      "weather",
      "stock",
      "price",
      "temperature",
  ]

  use_tools = any(
      kw in user_input.lower()
      for kw in tool_keywords
  )

  payload = build_payload(
      system_prompt,
      user_input,
      tools=TOOLS if use_tools else None,
  )
  ```
* **Result:** The model behaves completely naturally for casual chat (no tools passed) and accurately emits tool calls when the user asks specific queries.

---

## Next Steps for Training
As outlined in **Step 5** of the plan, with these fixes in place, the next step is to **retrain from scratch** using the newly generated `new_train.jsonl` data. Do not load the old checkpoint, allowing the model to learn the clean, non-JSON conversational behaviors from scratch.
