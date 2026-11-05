\# Hybrid Log Classification (FastAPI + Regex + Sentence Transformers + LLM)

&#x20;

A log classification service. Upload a CSV of log messages and get the same file back with a category for every log. Each log is sent to the cheapest method that can handle it: regex rules for known formats, a trained classifier on sentence embeddings for the rest, and a Groq-hosted LLM for free-form logs from a legacy system.

&#x20;

\---

&#x20;

\## Why hybrid

&#x20;

| Method | Good at | Weak at |

|---|---|---|

| Regex rules | Very fast, exact, free. Ideal for log formats you already know. | Only matches the patterns someone wrote. |

| Sentence Transformer + Logistic Regression | Understands wording variations. Fast and cheap once trained. | Needs labeled training data and can only answer with the classes it was trained on. |

| LLM (Groq) | Handles free-form text with no training data. | Slowest, costs quota, and answers need validation. |

&#x20;

Using the cheapest method that works keeps latency and cost low. The LLM is used only where the other two cannot work.

&#x20;

\---

&#x20;

\## Architecture

&#x20;

```mermaid

flowchart TD

&#x20;   U\["User uploads CSV<br/>(columns: source, log\_message)"] --> API\["FastAPI server<br/>POST /classify/"]

&#x20;   API --> V{"Valid CSV with the<br/>required columns?"}

&#x20;   V -- no --> ERR\["Error response"]

&#x20;   V -- yes --> R\["classify\_log()<br/>routes each row"]

&#x20;   R --> S{"source = LegacyCRM?"}

&#x20;   S -- yes --> LLM\["LLM via Groq<br/>Workflow Error / Deprecation Warning"]

&#x20;   S -- no --> RX\["Regex rules<br/>User Action / System Notification"]

&#x20;   RX --> M{"Rule matched?"}

&#x20;   M -- yes --> L1\["Label from regex"]

&#x20;   M -- no --> ML\["Sentence Transformer (all-MiniLM-L6-v2)<br/>+ Logistic Regression"]

&#x20;   ML --> C{"Confidence at least 50%?"}

&#x20;   C -- yes --> L2\["One of 5 classes"]

&#x20;   C -- no --> UN\["Unclassified"]

&#x20;   LLM --> OUT

&#x20;   L1 --> OUT

&#x20;   L2 --> OUT

&#x20;   UN --> OUT

&#x20;   OUT\["Add target\_label column<br/>save resources/output.csv"] --> RESP\["Return labeled CSV"]

```

&#x20;

\### Routing rules

&#x20;

1\. If `source` is `LegacyCRM`, the log goes straight to the LLM.

2\. Otherwise the regex rules are tried in order and the first match wins.

3\. If no rule matches, the message is embedded with `all-MiniLM-L6-v2` and the classifier returns a probability for each class.

4\. If the highest probability is below 0.5, the label is `Unclassified`.

| Tier | Used for | Labels it can produce |

|---|---|---|

| Regex | Known formats from every source except LegacyCRM | User Action, System Notification |

| Classifier | Logs the regex did not recognize | Critical Error, Error, HTTP Status, Resource Usage, Security Alert (or Unclassified) |

| LLM | All LegacyCRM logs | Workflow Error, Deprecation Warning (or Unclassified) |

&#x20;

\---

&#x20;

\## Tech stack

&#x20;

\- Python 3.11

\- FastAPI and Uvicorn

\- Sentence Transformers (`all-MiniLM-L6-v2`, pre-trained) and scikit-learn (Logistic Regression)

\- Groq API for the LLM (`openai/gpt-oss-120b`)

\- pandas, joblib, python-dotenv

\---

&#x20;

\## Project structure

&#x20;

```

server.py                          FastAPI app and the /classify/ endpoint

classify.py                        Routes each log to regex, the classifier or the LLM

processor\_regex.py                 Regex rules

processor\_bert.py                  Loads the embedding model and the saved classifier

processor\_llm.py                   Groq LLM call and answer parsing

requirements.txt                   Pinned dependencies

models/log\_classifier.joblib       Trained Logistic Regression classifier

training/log\_classification.ipynb  Notebook that trained the classifier

training/dataset/synthetic\_logs.csv  Labeled dataset (2,410 rows)

resources/test.csv                 Sample input

resources/output.csv               Result written by the server (overwritten on every request)

```

&#x20;

\---

&#x20;

\## Dataset and training

&#x20;

`training/dataset/synthetic\_logs.csv` has 2,410 synthetic logs with the columns `timestamp`, `source`, `log\_message`, `target\_label` and `complexity`. The `complexity` column marks the tier each log was designed for: 500 `regex`, 1,903 `bert` and 7 `llm`.

&#x20;

```mermaid

flowchart LR

&#x20;   D\["synthetic\_logs.csv<br/>2,410 labeled logs"] --> F\["Rows used by the classifier<br/>(1,903)"]

&#x20;   F --> EMB\["Embed messages<br/>all-MiniLM-L6-v2"]

&#x20;   EMB --> SPLIT\["70 / 30 split<br/>random\_state=42"]

&#x20;   SPLIT --> LR\["Train Logistic Regression"]

&#x20;   SPLIT --> EV\["Evaluate on held-out 30%<br/>(571 logs)"]

&#x20;   LR --> MOD\["models/log\_classifier.joblib"]

&#x20;   MOD -. loaded at server start .-> SRV\["processor\_bert.py"]

```

&#x20;

The saved model was fitted on the training 70% only, so the 571 test logs were never seen during training.

&#x20;

\---

&#x20;

\## Setup

&#x20;

\### Prerequisites

&#x20;

\- Python 3.11 (3.10 also works). Do not use 3.13: the pinned pandas version has no build for it.

\- Git

\- A free Groq API key from console.groq.com (needed for the LLM path)

\- About 3 GB of free disk space (PyTorch is large)

\### Install and run (Windows PowerShell)

&#x20;

```powershell

git clone https://github.com/hemanth10718/Hybrid-Log-CS.git

cd Hybrid-Log-CS\\nlp-log-classification\\project-nlp-log-classification-main\\project-nlp-log-classification-main

&#x20;

py -3.11 -m venv C:\\lcenv

C:\\lcenv\\Scripts\\python.exe -m pip install --upgrade pip

C:\\lcenv\\Scripts\\python.exe -m pip install -r requirements.txt

&#x20;

Set-Content -Path .env -Value "GROQ\_API\_KEY=your\_key\_here"

&#x20;

C:\\lcenv\\Scripts\\python.exe -m uvicorn server:app --reload

```

&#x20;

Notes:

&#x20;

\- The virtual environment is created at a short path (`C:\\lcenv`) because the long project path plus PyTorch's deeply nested files can exceed the Windows 260-character path limit during installation.

\- Run the server from the folder that contains `server.py`, because the model path is relative.

\- The first start downloads the embedding model (about 90 MB), so it needs internet access.

\- `--reload` does not notice changes to `.env`. Restart the server after editing it.

\- On Linux or macOS: `python3.11 -m venv .venv`, activate it, `pip install -r requirements.txt`, then `uvicorn server:app --reload`.

\### Configuration

&#x20;

Create a `.env` file in the folder that contains `server.py`:

&#x20;

```

GROQ\_API\_KEY=your\_key\_here

```

&#x20;

Never commit `.env` (it is listed in `.gitignore`). The LLM model name is set in `processor\_llm.py`. Groq retires models from time to time. If a request fails with `model\_decommissioned`, choose a current model from the Groq docs and update the name there.

&#x20;

\---

&#x20;

\## Usage

&#x20;

Open `http://127.0.0.1:8000/docs`, open \*\*POST /classify/\*\*, click \*\*Try it out\*\*, choose a CSV and click \*\*Execute\*\*. The API root has no page of its own, so a 404 at `/` is expected.

&#x20;

\*\*Input:\*\* a `.csv` file with the columns `source` and `log\_message`.

&#x20;

\*\*Output:\*\* the same table with an added `target\_label` column. It is also saved to `resources/output.csv` and overwritten on every request.

&#x20;

```bash

curl -X POST -F "file=@resources/test.csv" http://127.0.0.1:8000/classify/ -o labeled.csv

```

&#x20;

On Windows PowerShell use `curl.exe` instead of `curl`.

&#x20;

Example result from `resources/test.csv`:

&#x20;

| source | log\_message (shortened) | target\_label |

|---|---|---|

| ModernCRM | IP 192.168.133.114 blocked due to potential attack | Security Alert |

| AnalyticsEngine | Backup completed successfully. | System Notification |

| ModernHR | GET /v2/.../servers/detail HTTP/1.1 RCODE 200 ... | HTTP Status |

| LegacyCRM | Case escalation for ticket ID 7324 failed because the agent is no longer active. | Workflow Error |

| LegacyCRM | The 'ReportGenerator' module will be retired in version 4.0 ... | Deprecation Warning |

&#x20;

\---

&#x20;

\## Evaluation

&#x20;

The notebook evaluates the classifier on the held-out 30% (571 logs from a \*\*synthetic\*\* dataset):

&#x20;

| Class | Precision | Recall | F1 | Test logs |

|---|---|---|---|---|

| Critical Error | 0.91 | 1.00 | 0.95 | 48 |

| Error | 0.98 | 0.89 | 0.93 | 47 |

| HTTP Status | 1.00 | 1.00 | 1.00 | 304 |

| Resource Usage | 1.00 | 1.00 | 1.00 | 49 |

| Security Alert | 1.00 | 0.99 | 1.00 | 123 |

| \*\*Accuracy\*\* | | | \*\*0.99\*\* | 571 |

| \*\*Macro average\*\* | 0.98 | 0.98 | 0.98 | 571 |

&#x20;

How to read these numbers:

&#x20;

\- The data is synthetic and templated, so near-perfect scores say little about real-world logs.

\- `HTTP Status` makes up 304 of the 571 test logs (53%), which flatters accuracy. Macro F1, where every class counts equally, is the fairer headline.

\- Critical Error precision (0.91) and Error recall (0.89) are the weakest spots.

`evaluate\_logs.py` rebuilds the notebook's test split and also scores the \*\*full pipeline\*\* (regex first, then the classifier with the 50% rule) and the regex-routed logs. Run it from the folder that contains `classify.py`:

&#x20;

```powershell

C:\\lcenv\\Scripts\\python.exe evaluate\_logs.py

```

&#x20;

The LLM path cannot be measured meaningfully with the current data, which contains only 7 LegacyCRM logs.

&#x20;

\---

&#x20;

\## Troubleshooting

&#x20;

| Symptom | Cause | Fix |

|---|---|---|

| `GroqError: The api\_key client option must be set` | `GROQ\_API\_KEY` is missing | Create the `.env` file and restart |

| `401 Invalid API Key` | Wrong or deleted key, or the running server still holds the old one | Fix `.env` and restart the server |

| `model\_decommissioned` | Groq retired the model | Change the model name in `processor\_llm.py` |

| `TypeError ... unexpected keyword argument 'proxies'` | Old `groq` library with a newer `httpx` | Keep `httpx==0.27.2` (pinned in `requirements.txt`) |

| `ValueError: numpy.dtype size changed` | pandas built for NumPy 1.x but NumPy 2 installed | Keep `numpy<2` (pinned) |

| `OSError ... torch\\include\\...` during install | Windows 260-character path limit | Create the virtual environment at a short path such as `C:\\lcenv` |

| pandas fails to install | Python 3.13 | Use Python 3.11 |

&#x20;

\---

&#x20;

\## Limitations and future work

&#x20;

\- \*\*The classifier only knows five classes.\*\* Logs that belong to User Action, System Notification, Workflow Error or Deprecation Warning but are missed by the regex are forced into one of the five. For example, `User 12345 logged in.` is labeled Security Alert. An "Other" class or an LLM fallback would fix this.

\- \*\*Unclassified is final.\*\* Low-confidence logs are not sent to the LLM.

\- \*\*Routing is hard-coded to the source name `LegacyCRM`.\*\* A config list would be easier to maintain.

\- \*\*One LLM call per log\*\*, which is slow and costly on large files. Caching, batching or parallel calls would help.

\- \*\*LLM output is not validated\*\* against the allowed labels, and log text is placed directly in the prompt.

\- \*\*No error handling around the LLM call\*\*, so one failed call fails the whole upload.

\- \*\*The server writes to one shared output file\*\*, so concurrent requests overwrite each other.

\- Only 7 LegacyCRM logs exist, so that path is not measured. Hand-labeling 30 to 40 more would allow a real evaluation.

\- No authentication, automated tests or Docker setup yet.

\---

&#x20;

\## Security notes

&#x20;

\- Keep API keys in `.env` only. Never commit them, paste them into chats, or include them in screenshots.

\- Use a separate Groq key per project so one can be deleted without affecting the others.

\- If a key is ever exposed, delete it in the Groq console and create a new one.

\---

&#x20;

## Author

Developed by Hemanth R

## License

This project is licensed under the **MIT License**.

