# Application Setup

## Prerequisites
Ensure you have Docker and Docker Compose installed on your machine. You can follow the official installation guides if needed:
- [Docker Installation](https://docs.docker.com/get-docker/)
- [Docker Compose Installation](https://docs.docker.com/compose/install/)

## Environment Variables
Before running the application, set the following environment variables in a `.env` file in the parent directory:

### Steps to Create `.env` File:
1. In the **root directory** of the each, create a new file named `.env`.
2. Add the following environment variables to the `.env` file:

```plaintext
DB_HOST=
DB_NAME=
DB_USER=
DB_PASSWORD=
```

# Running the Application

## Create and Configure the .env File:

Follow the steps above to create the `.env` file and fill it with the appropriate values for your database connection.

## Start the Application:

1. In your terminal, navigate to the project directory.
2. Run the following command to start the application using Docker Compose:

   ```bash
   docker-compose up```

# Access the Application

Once the application is up and running, you can access it at [http://localhost:8080](http://localhost:8080).

## API Endpoints

### Upload Endpoint:
- **URL**: `http://localhost:8080/upload`
- **Method**: `POST`
- **Description**: Upload a CSV file to be processed, along with required form data (`marks`, `date`, `institute_id`, `class_id`).

#### Example Request:
Use tools like Postman or curl to make a request to the upload endpoint with the necessary form data and a CSV file.

```bash
curl -X 'POST' \
  'http://localhost:8080/upload' \
  -F 'marks_out_of=100' \
  -F 'date=2024-11-24' \
  -F 'institute_id=1' \
  -F 'class_id=2' \
  -F 'csv=@path/to/your/file.csv'
````

## Response

Upon successful upload, the API will return a message like this:

```json
{
  "message": "Data uploaded successfully"
}
```

#### Troubleshooting

If you encounter any issues, ensure that the database connection details in the `.env` file are correct.

Check the logs by running `docker-compose logs` for more information about the application status.

#### Additional Information

The application utilizes PostgreSQL as the database, which is configured through the environment variables.
You can customize the `docker-compose.yml` file to suit your environment if needed.
