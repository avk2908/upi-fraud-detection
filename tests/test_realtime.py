from fastapi.testclient import TestClient
from backend.app import app

def test_websocket_route_registered():
    assert "/ws/transactions" in {route.path for route in app.routes}

def test_simulator_pushes_a_scored_event():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transactions") as socket:
            client.post("/api/simulator/config",json={"rate":10})
            client.post("/api/simulator/start")
            message=socket.receive_json()
            if message.get("event")=="status": message=socket.receive_json()
            assert message["event"]=="transaction"
            assert "risk_score" in message["data"]
            client.post("/api/simulator/stop")
