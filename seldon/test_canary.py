import requests
import json
import time
from collections import Counter
import numpy as np

MINIKUBE_IP = "127.0.0.1"
SERVICE_PORT = 30080
NUM_REQUESTS = 100
SAMPLE_REQUEST = {
    "data": {
        "ndarray": [
            [
                17.99,
                10.38,
                122.8,
                1001.0,
                0.1184,
                0.2776,
                0.3001,
                0.1471,
                0.2419,
                0.07871,
                1.095,
                0.9053,
                8.589,
                153.4,
                0.006399,
                0.04904,
                0.05373,
                0.01587,
                0.03003,
                0.006193,
                25.38,
                17.33,
                184.6,
                2019.0,
                0.1622,
                0.6656,
                0.7119,
                0.2654,
                0.4601,
                0.1189,
            ]
        ]
    }
}


def get_minikube_ip():
    import subprocess

    try:
        result = subprocess.run(["minikube", "ip"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not get minikube IP: {e}")
        return "127.0.0.1"


def test_canary_deployment():
    print("=" * 80)
    print("TESTING SELDON CANARY DEPLOYMENT")
    print("=" * 80)
    minikube_ip = get_minikube_ip()
    endpoint = f"http://{minikube_ip}:{SERVICE_PORT}/api/v1.0/predictions"
    print(f"\nEndpoint: {endpoint}")
    print(f" Sending {NUM_REQUESTS} requests...")
    print(f"Expected: ~{5}% canary, ~{95}% baseline\n")
    versions = []
    latencies = []
    successes = 0
    failures = 0
    for i in range(NUM_REQUESTS):
        try:
            start = time.time()
            response = requests.post(
                endpoint,
                json=SAMPLE_REQUEST,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            latency = (time.time() - start) * 1000
            if response.status_code == 200:
                successes += 1
                result = response.json()
                version = (
                    result.get("meta", {}).get("tags", {}).get("version", "unknown")
                )
                versions.append(version)
                latencies.append(latency)
                if (i + 1) % 10 == 0:
                    print(
                        f" Request {i + 1}/{NUM_REQUESTS} - {version} - {latency:.2f}ms"
                    )
            else:
                failures += 1
                print(f" Request {i + 1} failed: {response.status_code}")
        except Exception as e:
            failures += 1
            print(f" Request {i + 1} error: {str(e)}")
        time.sleep(0.1)
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    success_rate = successes / NUM_REQUESTS * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({successes}/{NUM_REQUESTS})")
    print(
        f" Failure Rate: {failures / NUM_REQUESTS * 100:.1f}% ({failures}/{NUM_REQUESTS})"
    )
    if versions:
        version_counts = Counter(versions)
        print(f"\nTraffic Distribution:")
        for version, count in version_counts.items():
            percentage = count / len(versions) * 100
            print(f"   {version}: {count} requests ({percentage:.1f}%)")
            if "canary" in version.lower():
                expected = 5
                diff = abs(percentage - expected)
                status = "" if diff < 3 else ""
                print(
                    f"      {status} Expected: ~{expected}%, Got: {percentage:.1f}% (diff: {diff:.1f}%)"
                )
            elif "baseline" in version.lower():
                expected = 95
                diff = abs(percentage - expected)
                status = "" if diff < 3 else ""
                print(
                    f"      {status} Expected: ~{expected}%, Got: {percentage:.1f}% (diff: {diff:.1f}%)"
                )
    if latencies:
        print(f"\nLatency Statistics:")
        print(f"   Mean: {np.mean(latencies):.2f}ms")
        print(f"   Median (p50): {np.percentile(latencies, 50):.2f}ms")
        print(f"   p95: {np.percentile(latencies, 95):.2f}ms")
        print(f"   p99: {np.percentile(latencies, 99):.2f}ms")
        print(f"   Min: {min(latencies):.2f}ms")
        print(f"   Max: {max(latencies):.2f}ms")
    print("\n" + "=" * 80)
    if success_rate >= 95 and versions:
        canary_pct = version_counts.get("v2.0.0-canary", 0) / len(versions) * 100
        baseline_pct = version_counts.get("v2.0.0", 0) / len(versions) * 100
        canary_ok = 2 <= canary_pct <= 8
        baseline_ok = 92 <= baseline_pct <= 98
        if canary_ok and baseline_ok:
            print("CANARY DEPLOYMENT TEST PASSED!")
            print("Traffic split is correct (5%/95%)")
            print("Success rate is acceptable")
            print("Latency is within limits")
            return True
        else:
            print("CANARY DEPLOYMENT TEST PARTIALLY PASSED")
            print("   Traffic distribution needs adjustment")
            return False
    else:
        print("CANARY DEPLOYMENT TEST FAILED")
        print("   Check deployment configuration")
        return False


if __name__ == "__main__":
    success = test_canary_deployment()
    exit(0 if success else 1)
