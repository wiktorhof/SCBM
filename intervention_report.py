# import wandb 
# import wandb_workspaces.reports.v2 as wr

# # Create a basic report
# report = wr.Report(
#     entity="wiktorh",
#     project="PSCBM", 
#     title="My Programmatic Report",
#     description="A report created via API"
# )

# # Create the filters as a string expression
# filters_str = 'state != "crashed"'  and "config.model.model" == "pscbm"'# and Config("model.cov_type") == "amortized" and Config("model.concept_learning") == "hard" and Config("data") == "CUB" and (epoch <= 200 or epoch != None)'

# amortized_runset = wr.Runset(
#     entity="wiktorh",
#     project="PSCBM",
#     name="amortized_covariance",
#     filters=filters_str,
#     groupby=[
#         "config.model.mask_density",
#         "config.model.num_masks_train",
#         "epoch",
#     ],
# )


# # Training metrics (accuracy, loss) on training and validation sets
# panel_grid_training_metrics = wr.PanelGrid(
#     runsets=[amortized_runset],
#     # panels=[
#     #     wr.LinePlot(x="epoch", y="train/total_loss")
#     # ]
# )
# # Intervention vurves at different epochs
# panel_grid_intervention_curves= wr.PanelGrid(
#     runsets=[amortized_runset],
# )
# report.blocks = [
#     wr.H1("Global & amortized covariance"),
#     wr.P("This report compares the global and amortized covariance matrices for traning parameters and intervention curves throughout training."),
#     panel_grid_training_metrics,
#     panel_grid_intervention_curves,
# ]

# # Save the report (uploads to W&B server)
# report.save()


import wandb_workspaces.reports.v2 as wr

report = wr.Report(
     entity="entity",
     project="project",
     title="An amazing title",
     description="A descriptive description.",
)

blocks = [
     wr.PanelGrid(
         panels=[
             wr.LinePlot(x="time", y="velocity"),
             wr.ScatterPlot(x="time", y="acceleration"),
         ]
     )
]

report.blocks = blocks
report.save()
