from ghostfighter.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main([
            "all",
            "--out",
            "runs/default",
            "--episodes-per-style",
            "80",
            "--epochs",
            "8",
            "--eval-episodes",
            "160",
            "--stress",
        ])
    )
