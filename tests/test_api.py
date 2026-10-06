from fastapi.testclient import TestClient
from backend.app import app, service

def test_api_routes_registered():
    paths={route.path for route in app.routes}
    assert "/api/health" in paths and "/api/transactions/score" in paths

def test_health_schema():
    response=TestClient(app).get("/api/health")
    assert response.status_code==200
    assert {"status","models","simulator","mode"}.issubset(response.json())

def test_missing_graph_model_is_reported_as_degraded():
    model, status = service.hgnn_model, service.status.get("hgnn")
    service.hgnn_model=None
    service.status["hgnn"]="unavailable: test missing model"
    try:
        payload={"transaction_id":"missing-graph-test","type":"TRANSFER","amount":100,"oldbalanceOrg":100,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":100}
        result=TestClient(app).post("/api/transactions/score",json=payload).json()
        assert result["mode"]=="DEGRADED_MODEL"
        assert result["component_scores"]["hgnn"] is None
        assert "hgnn" in result["model_errors"]
    finally:
        service.hgnn_model=model
        service.status["hgnn"]=status
