import fastapi
import classifier
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClusterRequest(BaseModel):
    cluster_count: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/run_classifier")
def run_classifier(cluster_request: ClusterRequest):
    cluster_count = cluster_request.cluster_count
    classifier.run_classifier(cluster_count)
    return {"Classifier": "Executed"}
