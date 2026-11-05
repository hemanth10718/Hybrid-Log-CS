# Hybrid Log Classification (FastAPI + Regex + Sentence Transformers + LLM)

A log classification service. Upload a CSV of log messages and get the same file back with a category for every log. Each log is sent to the cheapest method that can handle it: regex rules for known formats, a trained classifier on sentence embeddings for the rest, and a Groq-hosted LLM for free-form logs from a legacy system.

---

## Why hybrid

| Method | Good at | Weak at |
|---|---|---|
| Regex rules | Very fast, exact, free. Ideal for log formats you already know. | Only matches the patterns someone wrote. |
| Sentence Transformer + Logistic Regression | Understands wording variations. Fast and cheap once trained. | Needs labeled training data and can only answer with the classes it was trained on. |
| LLM (Groq) | Handles free-form text with no training data. | Slowest, costs quota, and answers need validation. |

Using the cheapest method that works keeps latency and cost low. The LLM is used only where the other two cannot work.

---

## Architecture

```mermaid
flowchart TD
    U["User uploads CSV<br/>(columns: source, log_message)"] --> API["FastAPI server<br/>POST /classify/"]
    API --> V{"Valid CSV with the<br/>required columns?"}
    V -- no --> ERR["Error response"]
    V -- yes --> R["classify_log()<br/>routes each row"]
    R --> S{"source = LegacyCRM?"}
    S -- yes --> LLM["LLM via Groq<br/>Workflow Error / Deprecation Warning"]
    S -- no --> RX["Regex rules<br/>User Action / System Notification"]
    RX --> M{"Rule matched?"}
    M -- yes --> L1["Label from regex"]
    M -- no --> ML["Sentence Transformer (all-MiniLM-L6-v2)<br/>+ Logistic Regression"]
    ML --> C{"Confidence at least 50%?"}
    C -- yes --> L2["One of 5 classes"]
    C -- no --> UN["Unclassified"]
    LLM --> OUT
    L1 --> OUT
    L2 --> OUT
    UN --> OUT
    OUT["Add target_label column<br/>save resources/output.csv"] --> RESP["Return labeled CSV"]   
