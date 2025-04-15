#!/bin/bash
# Start FastAPI in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Handle shutdown signals
trap "kill $FASTAPI_PID $STREAMLIT_PID; exit" SIGINT SIGTERM

# Keep container running
wait $FASTAPI_PID $STREAMLIT_PID
