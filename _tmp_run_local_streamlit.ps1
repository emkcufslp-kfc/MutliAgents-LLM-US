$env:PYTHONPATH = 'D:\Codex projects\Multi LLM\TradingAgents\us_hedgefund_agents\.deps'
Set-Location 'D:\Codex projects\Multi LLM\TradingAgents\us_hedgefund_agents'
python -m streamlit run frontend/app.py --global.developmentMode false --server.headless true --server.port 8512
