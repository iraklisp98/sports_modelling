import unittest
from pathlib import Path

import yaml

from pipeline.run_pipeline import select_stages


ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
ROOT_COMPOSE = ROOT / "docker-compose.yml"


class DockerConfigTests(unittest.TestCase):
    def test_pipeline_orchestrator_lists_stages_in_order(self):
        stages = [stage.name for stage in select_stages()]

        self.assertEqual(
            stages,
            [
                "stage1_ingest",
                "stage2_features",
                "stage3_train",
                "poisson_goal_model",
                "stage4_odds_gen",
                "stage5_compare",
                "market_baseline_diagnostics",
                "model_diagnostics",
                "export_dashboard_data",
            ],
        )

    def test_pipeline_orchestrator_can_select_stage_slice(self):
        stages = select_stages(from_stage="stage3_train", to_stage="stage5_compare")

        self.assertEqual([stage.name for stage in stages], ["stage3_train", "poisson_goal_model", "stage4_odds_gen", "stage5_compare"])
        self.assertIn("--tracking-uri", stages[0].args)
        self.assertIn("file:mlruns", stages[2].args)

    def test_root_compose_supports_plain_docker_compose_up(self):
        compose = yaml.safe_load(ROOT_COMPOSE.read_text(encoding="utf-8"))
        services = compose["services"]

        self.assertEqual(services["pipeline"]["build"]["context"], ".")
        self.assertEqual(services["pipeline"]["build"]["dockerfile"], "docker/Dockerfile.pipeline")
        self.assertEqual(services["dashboard"]["build"]["context"], ".")
        self.assertEqual(services["dashboard"]["build"]["dockerfile"], "docker/Dockerfile.dashboard")
        self.assertIn("./data:/app/data", services["pipeline"]["volumes"])
        self.assertIn("./dashboard/data:/usr/share/nginx/html/data:ro", services["dashboard"]["volumes"])
        self.assertNotIn("./mlruns:/app/mlruns", services["pipeline"]["volumes"])
        self.assertEqual(services["pipeline"]["environment"]["MPLCONFIGDIR"], "/tmp/matplotlib")
        self.assertEqual(services["pipeline"]["environment"]["MLFLOW_ALLOW_FILE_STORE"], "true")

    def test_dockerfiles_define_pipeline_and_dashboard_runtime(self):
        pipeline = (DOCKER / "Dockerfile.pipeline").read_text(encoding="utf-8")
        dashboard = (DOCKER / "Dockerfile.dashboard").read_text(encoding="utf-8")
        nginx = (DOCKER / "nginx.conf").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.12-slim-bookworm", pipeline)
        self.assertIn("default-jre-headless", pipeline)
        self.assertIn("MLFLOW_ALLOW_FILE_STORE=true", pipeline)
        self.assertIn("mkdir -p /app/mlruns /tmp/matplotlib /tmp/home", pipeline)
        self.assertIn("pip install --no-cache-dir -r requirements.txt", pipeline)
        self.assertIn('CMD ["python", "pipeline/run_pipeline.py"]', pipeline)
        self.assertIn("FROM nginx:1.27-alpine", dashboard)
        self.assertIn("COPY dashboard/ /usr/share/nginx/html/", dashboard)
        self.assertIn("try_files $uri $uri/ /index.html", nginx)
        self.assertIn("location /data/", nginx)

    def test_compose_wires_batch_pipeline_and_static_dashboard(self):
        compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]

        self.assertIn("pipeline", services)
        self.assertIn("dashboard", services)
        self.assertEqual(services["dashboard"]["ports"], ["8080:80"])
        self.assertEqual(services["pipeline"]["user"], "${UID:-1000}:${GID:-1000}")
        self.assertEqual(services["dashboard"]["depends_on"]["pipeline"]["condition"], "service_completed_successfully")
        self.assertIn("../data:/app/data", services["pipeline"]["volumes"])
        self.assertIn("../dashboard/data:/app/dashboard/data", services["pipeline"]["volumes"])
        self.assertNotIn("../mlruns:/app/mlruns", services["pipeline"]["volumes"])
        self.assertEqual(services["pipeline"]["environment"]["MLFLOW_TRACKING_URI"], "file:/app/mlruns")
        self.assertEqual(services["pipeline"]["environment"]["MLFLOW_ALLOW_FILE_STORE"], "true")
        self.assertEqual(services["pipeline"]["environment"]["HOME"], "/tmp/home")
        self.assertIn("../dashboard/data:/usr/share/nginx/html/data:ro", services["dashboard"]["volumes"])

    def test_dockerignore_excludes_large_and_secret_paths(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

        for path in [".env", "data", "mlruns", "dashboard/data", ".venv"]:
            self.assertIn(path, dockerignore)


if __name__ == "__main__":
    unittest.main()
