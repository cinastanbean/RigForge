from fastapi.testclient import TestClient

from rigforge.main import app


def test_cpu_preference_change_refreshes_build_in_same_session():
    client = TestClient(app)
    session_id = "s-live-refresh-intel-switch"

    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "预算10000，2K游戏，不需要显示器，1TB，静音，AMD",
            "enthusiasm_level": "standard",
            "build_data_mode": "jd_newegg",
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["requirements"]["cpu_preference"] == "AMD"
    assert first_payload["build"]["cpu"] is not None
    assert first_payload["build"]["cpu"]["brand"].lower() == "amd"

    second = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Intel",
            "enthusiasm_level": "standard",
            "build_data_mode": "jd_newegg",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["requirements"]["cpu_preference"] == "Intel"
    assert second_payload["build"]["cpu"] is not None
    assert second_payload["build"]["cpu"]["brand"].lower() == "intel"


def test_lowercase_intel_never_falls_back_to_amd_in_same_session():
    client = TestClient(app)
    session_id = "s-live-refresh-lowercase-intel"

    # 使用规则模式避免 LLM 行为不稳定
    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "预算10000，2K游戏，不需要显示器，1TB，静音，amd",
            "enthusiasm_level": "standard",
            "build_data_mode": "jd",
            "model_provider": "rules",  # 使用规则模式
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["requirements"]["cpu_preference"] == "AMD"

    second = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "intel",
            "enthusiasm_level": "standard",
            "build_data_mode": "jd",
            "model_provider": "rules",  # 使用规则模式
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["requirements"]["cpu_preference"] == "Intel"
    cpu = second_payload["build"]["cpu"]
    assert cpu is not None
    assert cpu["brand"].lower() in ("intel", "英特尔")
