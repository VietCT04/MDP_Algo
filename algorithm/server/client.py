import requests

BASE = "http://127.0.0.1:5000"

def post(p, j): return requests.post(BASE + p, json=j).json()
def get(p):     return requests.get(BASE + p).json()

print(post("/car/initialize", {"x": 20, "y": 20, "theta": 0.0}))

obs = [
    {"x": 60, "y": 120, "image_side": "N"},
    {"x": 140, "y": 80, "image_side": "E"},
]
for o in obs:
    print(post("/obstacles/add", o))

print(post("/mission/plan", {"targets": [0, 1]}))

for i in range(20):
    nxt = get("/car/next_action")
    print("NEXT:", nxt)
    if nxt.get("status") != "success":
        break
    action = nxt["action"]
    params = nxt["parameters"]
    expected = nxt["expected_end_state"]
    res = post("/car/execute", {
        "action": action,
        "parameters": params,
        "actual_result": {"measured_position": expected}
    })
    print("EXEC:", res)
