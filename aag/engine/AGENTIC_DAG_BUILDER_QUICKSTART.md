# Agentic DAG Builder Quickstart

## 1) Activate conda environment

```powershell
conda activate AAG
```

## 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3) Enable new DAG builder path

Set feature flag before starting AAG engine:

```powershell
$env:AAG_USE_LANGGRAPH_DAG_BUILDER="1"
```

If you use CMD:

```cmd
set AAG_USE_LANGGRAPH_DAG_BUILDER=1
```

## 4) LLM provider recommendation

Default config now uses Ollama for DAG building:

- `config/engine_config.yaml`
- `reasoner.llm.provider: "ollama"`

Make sure your Ollama service is running and the configured model is available.

## 5) Runtime behavior

- When feature flag is ON, Scheduler first tries LangGraph DAG Builder.
- If LangGraph path fails, Scheduler automatically falls back to legacy DAG build logic.

This allows safe incremental rollout and quick rollback.

## 6) Rollback

Disable feature flag to return to legacy path:

```powershell
Remove-Item Env:AAG_USE_LANGGRAPH_DAG_BUILDER
```
