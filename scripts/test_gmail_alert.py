"""
scripts/test_gmail_alert.py
CLI utility to test Gmail SMTP credentials and Prometheus Alertmanager webhook.

Usage:
  1. Test direct Gmail SMTP connection:
     python scripts/test_gmail_alert.py --smtp --sender "user@gmail.com" --password "xxxx xxxx xxxx xxxx" --receiver "to@gmail.com"

  2. Test Alertmanager alert dispatch (requires docker-compose up):
     python scripts/test_gmail_alert.py --alertmanager
"""

import argparse
import json
import smtplib
import sys
import time
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def test_smtp_direct(sender: str, password: str, receiver: str):
    print(f"[*] Testing direct connection to smtp.gmail.com:587 for {sender}...")
    clean_pw = password.replace(" ", "")
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "[TEST] Real-Time IDS Gmail Alert Verification"
    msg["From"] = sender
    msg["To"] = receiver
    
    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; padding: 20px;">
        <div style="border-left: 4px solid #38a169; background: #f0fff4; padding: 15px; border-radius: 4px;">
          <h2 style="color: #22543d; margin-top: 0;">✅ Gmail Alerting Verified</h2>
          <p>Your Gmail credentials for the <strong>Real-Time IDS Prometheus Alertmanager</strong> are valid and operational.</p>
          <p>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.set_debuglevel(0)
            server.starttls()
            server.login(sender, clean_pw)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"[+] SUCCESS: Test email delivered successfully to {receiver}!")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"[-] AUTHENTICATION FAILED: {e}")
        print("    Ensure 2-Step Verification is active on Google, and use a 16-character App Password (not your primary password).")
        return False
    except Exception as e:
        print(f"[-] ERROR: {e}")
        return False


def test_alertmanager_webhook(url: str = "http://localhost:9093/api/v1/alerts"):
    print(f"[*] Dispatching test synthetic alert to Alertmanager at {url}...")
    
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = [
        {
            "labels": {
                "alertname": "TestCriticalAttackAlert",
                "severity": "critical",
                "component": "test_suite",
                "instance": "manual_trigger"
            },
            "annotations": {
                "summary": "Manual test alert from test_gmail_alert.py",
                "description": "Synthetic verification alert simulating a critical DoS/DDoS intrusion event."
            },
            "startsAt": now_iso
        }
    ]
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            if status in (200, 202):
                print(f"[+] SUCCESS: Alert accepted by Alertmanager (HTTP {status}). Check your Gmail inbox!")
                return True
            else:
                print(f"[-] Unexpected response from Alertmanager: HTTP {status} — {body}")
                return False
    except urllib.error.URLError as e:
        print(f"[-] Connection failed: {e}")
        print("    Ensure Docker monitoring stack is running: docker compose up -d")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Gmail Alerting for IDS Prometheus Service")
    parser.add_argument("--smtp", action="store_true", help="Test direct Gmail SMTP login and dispatch")
    parser.add_argument("--sender", type=str, help="Sender Gmail address")
    parser.add_argument("--password", type=str, help="16-character Google App Password")
    parser.add_argument("--receiver", type=str, help="Recipient email address")
    parser.add_argument("--alertmanager", action="store_true", help="Send test alert to Alertmanager endpoint")
    parser.add_argument("--url", type=str, default="http://localhost:9093/api/v1/alerts", help="Alertmanager API URL")
    
    args = parser.parse_args()
    
    if args.smtp:
        if not args.sender or not args.password or not args.receiver:
            print("Error: --smtp requires --sender, --password, and --receiver.")
            sys.exit(1)
        ok = test_smtp_direct(args.sender, args.password, args.receiver)
        sys.exit(0 if ok else 1)
    elif args.alertmanager:
        ok = test_alertmanager_webhook(args.url)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
