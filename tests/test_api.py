"""Endpoint schema and integration loop verifications.

Design ref: design-doc.md §5.1 — API Throughput Check: automated
integration checks querying endpoints with standard payloads, asserting
status code replies are strictly 200 OK.
"""


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
