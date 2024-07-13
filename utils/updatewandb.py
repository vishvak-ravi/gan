import wandb

api = wandb.Api()

# Access attributes directly from the run object
# or from the W&B App
username = "vishravi"
project = "gan"
run_id = "84cvmu05"

run = api.run(f"{username}/{project}/{run_id}")
run.config["dropout"] = 0
run.update()