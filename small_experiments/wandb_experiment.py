import wandb

print(f"wandb version: {wandb.__version__}")
# Initialize API and get runs
api = wandb.Api()
wandb_entity='wiktorh'
project_name='debug'
with wandb.init(project=project_name, reinit="create_new", entity=wandb_entity) as run:
    print (f"Run {run.name} initialized.")


    with wandb.init(project=project_name, reinit="create_new", entity=wandb_entity) as nested_run:
        print (f"Nested run's name: {nested_run.name}.")
        nested_run.log({"epoch": 1})
        run.log({"epoch": 2})
        print (f"Outer run's name: {run.name}.")

    run.log({"epoch": 3})
    print(f"After finishing inner run, the main run ({run.name}) is still active.")