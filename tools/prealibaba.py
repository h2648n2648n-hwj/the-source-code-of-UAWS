import pandas as pd
import os

def process_batch_task_data(input_file, output_dir, min_tasks=100):
    """
    Process the batch_task dataset by removing unnecessary columns, filtering out jobs
    with task counts less than the specified minimum, and saving each job's tasks
    as a separate CSV file.

    :param input_file: Path to the input CSV file
    :param output_dir: Path to the output directory where processed files will be saved
    :param min_tasks: Minimum number of tasks required per job to be included
    """
    # Read the input CSV file
    raw_data = pd.read_csv(input_file)

    # Filter the data to exclude unwanted rows, create a copy to avoid modifying the view
    filtered_data = raw_data[(raw_data['start_time'] > 86400) &
                             (raw_data['end_time'] > 0) &
                             (raw_data['status'] == 'Terminated')].copy()

    filtered_data['makespan'] = filtered_data.apply(lambda row: row['end_time'] - row['start_time'], axis=1)

    # Drop unnecessary columns
    columns_to_drop = ['instance_num', 'status', 'end_time', 'task_type', 'start_time', 'plan_cpu', 'plan_mem']
    filtered_data = filtered_data.drop(columns=columns_to_drop)

    # Group the data by job_name and count the number of tasks for each job
    job_task_counts = filtered_data.groupby('job_name').size()

    # Filter jobs that have more tasks than the specified minimum
    selected_jobs = job_task_counts[job_task_counts > min_tasks]

    # Retrieve the data for these selected jobs
    job_filtered_data = filtered_data[filtered_data['job_name'].isin(selected_jobs.index)]

    # Remove tasks whose task_name starts with "task_"
    job_filtered_data = job_filtered_data[~job_filtered_data['task_name'].str.startswith('task_')]

    # Get the unique job names
    unique_job_names = job_filtered_data['job_name'].unique()

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each job_name, filter the corresponding tasks and save them to separate CSV files
    for job_name in unique_job_names:
        job_data = job_filtered_data[job_filtered_data['job_name'] == job_name]

        # Define the file name, save it as a CSV file with the job_name
        output_file = os.path.join(output_dir, f"tasks_{job_name}.csv")

        # Save the tasks of this job to a CSV file
        job_data.to_csv(output_file, index=False)


# Main program execution
if __name__ == "__main__":
    input_file = "../workflows/alibaba/batch_task.csv"  # Path to the input file
    output_dir = "../workflows/alibaba/per_csv_300"  # Path to the output directory

    # Process the batch task data
    process_batch_task_data(input_file, output_dir, min_tasks=50)
