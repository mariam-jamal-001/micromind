import wandb

# Initialize the API
api = wandb.Api()

# Retrieve the specific run by name
run = api.run("mariam-jamal001-fhd/search_space_phinet/cosmic-universe-29/")

# for run in runs:
#     print(f"Run Name: {run.name}, Run ID: {run.id}")

# Retrieve the artifact associated with the run
artifact = api.artifact('run-8482yzbo-summary_table:v0')  # Replace 'table_name' with the actual artifact name

# Get the table from the artifact
table = artifact.get("summary_table")  # Replace "table_name" with the name you saved the table as

# Extract the data from the "num_params" column
num_params = table.get_column("num_params")

# Log a histogram for the distribution of parameters
wandb.init(project="search_space_phinet", entity="mariam-jamal001-fhd")
wandb.log({"Parameters Distribution": wandb.Histogram(num_params)})
wandb.finish()
