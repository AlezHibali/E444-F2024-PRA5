import pytest
import requests

BASE_URL = "http://ece-444-pra5-env-2.eba-ujsdbkn2.us-east-1.elasticbeanstalk.com/predict"

# Test data
test_data = [
    {"input": "This is fake news", "gt": "FAKE"},
    {"input": "This is a real data", "gt": "REAL"},
    {"input": "Politicians are lying to you", "gt": "FAKE"},
    {"input": "This is real data", "gt": "REAL"}
]

@pytest.mark.parametrize("test_case", test_data)
def test_prediction(test_case):
    response = requests.post(
        BASE_URL,
        json={"article": test_case["input"]},
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == test_case["gt"]

    response = requests.post(
        BASE_URL,
        json={"article":""},
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    
