def get_gcs_path_of_kaggle_data(data_name, is_private=False):
    if is_private:
        # Step 1: Get the credential from the Cloud SDK
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        user_credential = user_secrets.get_gcloud_credential()
        # Step 2: Set the credentials
        user_secrets.set_tensorflow_credential(user_credential)

    # Step 3: Use a familiar call to get the GCS path of the dataset
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path(data_name)	
    return GCS_DS_PATH 

