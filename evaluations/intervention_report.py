import wandb 
import wandb_workspaces.reports.v2 as wr

# Create a basic report
report = wr.Report(
    entity="wiktorh",
    project="PSCBM", 
    title="My Programmatic Report",
    description="A report created via API"
)

amortized_runset = wr.RunSet(
    entity="wiktorh",
    project="PSCBM",
    name="amortized_covariance",
    filters={
        "state": "finished",
        "config.model.model": "pscbm",
        "config.model.cov_type": "amortized",
        "config.model.concept_learning": "hard",
        "config.data": "CUB",
        "$or": [{"epoch": {"$lte": 200}}, {"epoch": {"$exists": False}}],
    },
    groupby=[
        "config.model.mask_density",
        "config.model.num_masks_train"
    ],
    order=[wr.OrderBy("created_at", ascending=False)],
    )

# Filtering for a missing field:
# "field": {"$exists": False}
# "field": {"$exists": True}
# "field": null
# "field": {"$ne": null}
# However the literal null is not recognized by Python.

# Training metrics (accuracy, loss) on training and validation sets
panel_grid_training_metrics = wr.PanelGrid(
    runsets=[runset],
    panels=...
)
# Intervention vurves at different epochs
panel_grid_intervention_curves= wr.PanelGrid(
    runsets=[runset],
    panels=...
)
report.blocks = [
    wr.H1("Global & amortized covariance"),
    wr.P("This report compares the global and amortized covariance matrices for traning parameters and intervention curves throughout training."),
    panel_grid,
]

# Save the report (uploads to W&B server)
report.save()