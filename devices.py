#!/usr/bin/env python3
"""Writes live_devices.json every few seconds with the devices on the hotspot.

Combines:
  - iw station dump      -> who is connected to Wi-Fi right now (+ signal, connected time)
  - dnsmasq.leases       -> IP address and device name handed out by DHCP
  - ip neigh             -> whether the device has answered recently
Run it from the IoT-Threat-Detection folder:  python3 devices.py
"""
import json
import os
import subprocess
import time

IFACE = "wlp3s0"
LEASES = "/var/lib/misc/dnsmasq.leases"
OUT = "live_devices.json"
INTERVAL = 3  # seconds


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def read_leases():
    leases = {}
    try:
        with open(LEASES) as f:
            for line in f:
                p = line.split()
                if len(p) >= 4:
                    leases[p[1].lower()] = {
                        "ip": p[2],
                        "name": "" if p[3] == "*" else p[3],
                        "lease_expires": int(p[0]),
                    }
    except OSError:
        pass
    return leases


def read_stations():
    stations, cur = {}, None
    for line in run(["iw", "dev", IFACE, "station", "dump"]).splitlines():
        line = line.strip()
        if line.startswith("Station"):
            cur = line.split()[1].lower()
            stations[cur] = {}
        elif cur and ":" in line:
            key, val = (x.strip() for x in line.split(":", 1))
            try:
                if key == "signal":
                    stations[cur]["signal_dbm"] = int(val.split()[0])
                elif key == "connected time":
                    stations[cur]["connected_s"] = int(val.split()[0])
                elif key == "inactive time":
                    stations[cur]["inactive_ms"] = int(val.split()[0])
            except ValueError:
                pass
    return stations


def read_neigh():
    neigh = {}
    for line in run(["ip", "neigh", "show", "dev", IFACE]).splitlines():
        p = line.split()
        if "lladdr" in p:
            mac = p[p.index("lladdr") + 1].lower()
            neigh[mac] = {"ip": p[0], "state": p[-1]}
    return neigh


def is_random_mac(mac):
    # Phones and PCs often use a "private" (randomised) Wi-Fi address
    return bool(int(mac.split(":")[0], 16) & 0x02)


def snapshot():
    leases, stations, neigh = read_leases(), read_stations(), read_neigh()
    devices = []
    for mac in set(leases) | set(stations) | set(neigh):
        lease, sta, nb = leases.get(mac, {}), stations.get(mac), neigh.get(mac, {})
        devices.append({
            "mac": mac,
            "ip": lease.get("ip") or nb.get("ip", ""),
            "name": lease.get("name", ""),
            "online": sta is not None or nb.get("state") in ("REACHABLE", "DELAY", "PROBE"),
            "signal_dbm": (sta or {}).get("signal_dbm"),
            "connected_s": (sta or {}).get("connected_s"),
            "arp_state": nb.get("state", ""),
            "private_mac": is_random_mac(mac),
        })
    devices.sort(key=lambda d: (not d["online"], d["ip"]))
    return {"updated": int(time.time()), "devices": devices}


def main():
    print(f"Writing {OUT} every {INTERVAL}s (Ctrl+C to stop)")
    while True:
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snapshot(), f, indent=2)
        os.replace(tmp, OUT)  # atomic, so the dashboard never reads half a file
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
