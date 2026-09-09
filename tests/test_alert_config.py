"""
tests/test_alert_config.py
Tests verifying the integrity and schema of:
  - alert_rules.yml
  - alertmanager/alertmanager.yml
  - prometheus.yml
  - docker-compose.yml
"""

import os
import unittest
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestAlertConfiguration(unittest.TestCase):
    def test_prometheus_config(self):
        prom_path = os.path.join(BASE_DIR, "prometheus.yml")
        self.assertTrue(os.path.exists(prom_path), "prometheus.yml must exist")
        
        with open(prom_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        self.assertIn("rule_files", cfg, "prometheus.yml must define rule_files")
        self.assertIn("/etc/prometheus/alert_rules.yml", cfg["rule_files"])
        self.assertIn("alerting", cfg, "prometheus.yml must define alerting section")
        self.assertIn("alertmanagers", cfg["alerting"])
        
        targets = cfg["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
        self.assertIn("alertmanager:9093", targets, "Alertmanager target missing from prometheus.yml")

    def test_alert_rules_syntax(self):
        rules_path = os.path.join(BASE_DIR, "alert_rules.yml")
        self.assertTrue(os.path.exists(rules_path), "alert_rules.yml must exist")
        
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
            
        self.assertIn("groups", rules, "alert_rules.yml must contain groups")
        self.assertGreater(len(rules["groups"]), 0)
        
        rule_group = rules["groups"][0]
        self.assertIn("rules", rule_group)
        
        alert_names = [r["alert"] for r in rule_group["rules"]]
        self.assertIn("IDSServiceDown", alert_names)
        self.assertIn("CriticalAttackDetected", alert_names)
        self.assertIn("HighAttackRateDetected", alert_names)
        
        for r in rule_group["rules"]:
            self.assertIn("alert", r, "Rule missing alert name")
            self.assertIn("expr", r, "Rule missing expression")
            self.assertIn("labels", r, "Rule missing labels")
            self.assertIn("severity", r["labels"], f"Rule {r['alert']} missing severity label")
            self.assertIn("annotations", r, f"Rule {r['alert']} missing annotations")
            self.assertIn("summary", r["annotations"], f"Rule {r['alert']} missing summary")

    def test_alertmanager_config(self):
        am_path = os.path.join(BASE_DIR, "alertmanager", "alertmanager.yml")
        self.assertTrue(os.path.exists(am_path), "alertmanager.yml must exist")
        
        with open(am_path, "r", encoding="utf-8") as f:
            am_cfg = yaml.safe_load(f)
            
        self.assertIn("global", am_cfg, "alertmanager.yml missing global block")
        self.assertEqual(am_cfg["global"].get("smtp_smarthost"), "smtp.gmail.com:587")
        self.assertIn("smtp_from", am_cfg["global"])
        self.assertIn("smtp_auth_username", am_cfg["global"])
        self.assertIn("smtp_auth_password", am_cfg["global"])
        self.assertTrue(am_cfg["global"].get("smtp_require_tls"))
        
        self.assertIn("route", am_cfg)
        self.assertIn("receiver", am_cfg["route"])
        
        receivers = am_cfg.get("receivers", [])
        self.assertGreater(len(receivers), 0)
        recv = receivers[0]
        self.assertIn("email_configs", recv)
        self.assertGreater(len(recv["email_configs"]), 0)

    def test_docker_compose_alertmanager_service(self):
        compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
        self.assertTrue(os.path.exists(compose_path), "docker-compose.yml must exist")
        
        with open(compose_path, "r", encoding="utf-8") as f:
            compose = yaml.safe_load(f)
            
        services = compose.get("services", {})
        self.assertIn("alertmanager", services, "alertmanager service missing in docker-compose.yml")
        self.assertIn("prometheus", services)
        
        am = services["alertmanager"]
        self.assertEqual(am["image"], "prom/alertmanager:latest")
        self.assertIn("9093:9093", am.get("ports", []))
        
        prom = services["prometheus"]
        self.assertIn("depends_on", prom)
        self.assertIn("alertmanager", prom["depends_on"])

    def test_docker_compose_test_and_live_variants(self):
        for variant in ["docker-compose.test.yml", "docker-compose.live.yml"]:
            compose_path = os.path.join(BASE_DIR, variant)
            self.assertTrue(os.path.exists(compose_path), f"{variant} must exist")
            with open(compose_path, "r", encoding="utf-8") as f:
                compose = yaml.safe_load(f)
            services = compose.get("services", {})
            self.assertIn("alertmanager", services, f"alertmanager missing in {variant}")
            self.assertIn("prometheus", services, f"prometheus missing in {variant}")
            self.assertIn("grafana", services, f"grafana missing in {variant}")

        test_dash_path = os.path.join(BASE_DIR, "grafana", "dashboards", "ids_test_dashboard.json")
        live_dash_path = os.path.join(BASE_DIR, "grafana", "dashboards", "ids_live_dashboard.json")
        self.assertTrue(os.path.exists(test_dash_path), "ids_test_dashboard.json must exist")
        self.assertTrue(os.path.exists(live_dash_path), "ids_live_dashboard.json must exist")

    def test_telegram_removed_from_config_and_code(self):
        import json
        config_path = os.path.join(BASE_DIR, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertNotIn("telegram", cfg, "config.json must not contain 'telegram' block")

        from alerts.alert_manager import AlertManager
        am = AlertManager(db_path=":memory:")
        self.assertFalse(hasattr(am, "_telegram_config"), "AlertManager must not have _telegram_config")
        self.assertFalse(hasattr(am, "_send_telegram_notification"), "AlertManager must not have _send_telegram_notification")

        # Verify zero occurrences in src/
        src_dir = os.path.join(BASE_DIR, "src")
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8", errors="ignore") as pf:
                        content = pf.read().lower()
                        self.assertNotIn("telegram", content, f"Found 'telegram' reference in {path}")

    def test_config_fail_fast_on_invalid_json(self):
        import tempfile
        from alerts.alert_rules import AlertRules
        from alerts.alert_manager import AlertManager

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write("{malformed json content: true,")
            bad_json_path = tf.name

        try:
            with self.assertRaises(ValueError):
                AlertRules(bad_json_path)

            with self.assertRaises(ValueError):
                AlertManager._load_dedup_window(bad_json_path)
        finally:
            if os.path.exists(bad_json_path):
                os.unlink(bad_json_path)


if __name__ == "__main__":
    unittest.main()


