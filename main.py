#!/usr/bin/env python3
"""
Interactive GitHub Activity CLI
Prompts for username and optional token, then fetches recent events.

Requirements: Only Python standard library.
"""

import json
import urllib.request
import urllib.error

API_URL = "https://api.github.com/users/{username}/events"


def fetch_events(username, token=None, timeout=10):
    """Fetch events for the given username. Returns a list of events."""
    url = API_URL.format(username=username)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "github-activity-cli/1.0")
    req.add_header("Accept", "application/vnd.github.v3+json")
    if token:
        req.add_header("Authorization", f"token {token}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body)
            message = err_json.get("message", err_body)
        except Exception:
            message = e.reason
        raise RuntimeError(f"HTTP {e.code}: {message}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def summarize_event(ev):
    """Turn GitHub event into a readable summary."""
    typ = ev.get("type", "UnknownEvent")
    repo = ev.get("repo", {}).get("name", "<unknown>")
    payload = ev.get("payload", {})

    if typ == "PushEvent":
        size = payload.get("size", 0)
        return f"Pushed {size} commit{'s' if size != 1 else ''} to {repo}"
    elif typ == "IssuesEvent":
        action = payload.get("action", "did something")
        return f"{action.capitalize()} an issue in {repo}"
    elif typ == "WatchEvent":
        return f"Starred {repo}"
    elif typ == "ForkEvent":
        return f"Forked {repo}"
    elif typ == "CreateEvent":
        ref_type = payload.get("ref_type")
        ref = payload.get("ref")
        return f"Created {ref_type} {ref} in {repo}" if ref_type and ref else f"Created {ref_type} in {repo}"
    return f"{typ} in {repo}"


def main():
    print("=== GitHub Activity CLI ===")
    username = input("Enter GitHub username: ").strip()
    if not username:
        print("Username is required.")
        return

    token = input("Enter your GitHub personal access token (optional): ").strip() or None

    try:
        events = fetch_events(username, token)
    except RuntimeError as e:
        print("Error:", e)
        return

    if not isinstance(events, list):
        print("Unexpected response:", events)
        return

    if not events:
        print(f"No recent public activity found for {username}.")
        return

    print(f"\nRecent activity for {username}:\n")
    for ev in events[:10]:  # Show first 10 events
        print(f"- {summarize_event(ev)}")


if __name__ == "__main__":
    main()
