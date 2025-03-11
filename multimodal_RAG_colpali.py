from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
import torch
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import pickle
import time
import matplotlib.pyplot as plt  # Add matplotlib for image display

class EmbedData:
    def __init__(self, embed_model_name="vidore/colpali-v1.2", batch_size=1):
        self.embed_model_name = embed_model_name
        self.embed_model, self.processor = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        print("\n=== Loading ColPali Model ===")
        device = "cpu"  # Force CPU
        print(f"Using device: {device}")
        
        print("Loading embed model from:", self.embed_model_name)
        embed_model = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        print("ColPali model loaded successfully")
        processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
        print("Processor loaded successfully")
        return embed_model, processor

    def embed(self, images):
        print("\n=== Starting Image Embedding Process ===")
        print(f"Total images to process: {len(images)}")
        self.embeddings = []
        for i, img in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}")
            print(f"Image size: {img.size}")
            inputs = self.processor.process_images([img]).to("cpu")
            print("Image processed by ColPali processor")
            
            with torch.no_grad():
                print("Running ColPali model inference...")
                outputs = self.embed_model(**inputs).cpu().numpy()
                print(f"Embedding shape: {outputs.shape}")
            
            self.embeddings.append(outputs[0])
            print(f"Embedding stored for image {i+1}")
        
        print(f"\nAll {len(self.embeddings)} images embedded successfully")

    def embed_query(self, query_text):
        print("\n=== Processing Query ===")
        print(f"Original query: {query_text}")
        try:
            query_with_token = "<image> " + query_text
            print(f"Query with token: {query_with_token}")
            
            print("Creating blank image for query...")
            blank_image = Image.new('RGB', (224, 224), color='white')
            
            print("Processing query through ColPali...")
            query_inputs = self.processor(
                text=query_with_token,
                images=[blank_image],
                return_tensors="pt"
            )
            print("Query processed by processor")
            
            query_inputs = {k: v.to("cpu") for k, v in query_inputs.items()}
            print("Inputs moved to CPU")
            
            with torch.no_grad():
                print("Running query through ColPali model...")
                query_emb = self.embed_model(**query_inputs).cpu().numpy()
                print(f"Query embedding shape: {query_emb.shape}")
            
            return query_emb
        except Exception as e:
            print(f"\nError in query embedding: {e}")
            print("Falling back to random embedding...")
            if len(self.embeddings) > 0:
                return np.random.rand(*self.embeddings[0].shape)
            else:
                return np.random.rand(1, 128)

# Convert PDF to images
pdf_path = "somatosensory.pdf"
embeddata = EmbedData()

# Add this section before PDF processing
pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
storage_dir = f"{pdf_name}_storage"

# For Qdrant Cloud
client = QdrantClient(
    url="https://294ea436-1b6c-4804-b7c0-ee36616a2e0b.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e585KbO5RoOGfHTNtVr86-CRvwSLFKiPRqyjsEnyttc",
    timeout=120.0  # Increase timeout to 120 seconds
)

# Check if embeddings are already cached
pickle_path = f"{pdf_name}_embeddings.pkl"
print("\n=== Checking Cache Status ===")
if os.path.exists(pickle_path) and client.collection_exists("pdf_docs"):
    print("Found cached embeddings")
    print(f"Loading from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        embeddata.embeddings = pickle.load(f)
    print("Embeddings loaded from cache")
    
    print("\n=== Verifying Qdrant Status ===")
    #print("Checking if collection exists:", client.collection_exists("pdf_docs"))
    if client.collection_exists("pdf_docs"):
        collection_info = client.get_collection("pdf_docs")
        #print("Collection info:", collection_info)
        # Get count of vectors in collection
        count = client.count("pdf_docs")
        #print("Number of vectors in collection:", count.count)
    
    print("\nLoading PDF images for GPT...")
    images = convert_from_path(pdf_path)
    print(f"Loaded {len(images)} pages from PDF")
else:
    print("No cache found - processing PDF from scratch")
    print("\nConverting PDF to images...")
    images = convert_from_path(pdf_path)
    print(f"Converted {len(images)} pages")
    
    print("\nStarting embedding process...")
    embeddata.embed(images)
    
    print("\nSaving embeddings to cache...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddata.embeddings, f)
    print("Embeddings cached successfully")
    
    print("\n=== Setting up Qdrant Collection ===")
    if not client.collection_exists("pdf_docs"):
        print("Creating new collection: pdf_docs")
        client.create_collection(
            collection_name="pdf_docs",
            vectors_config={
                "size": 128, 
                "distance": "Cosine",
                "multivector_config": {
                    "comparator": "max_sim"  # Find the best matching segment in each document
                }
            }
        )
        print("Collection created successfully")
    
    print("\n=== Uploading Embeddings to Qdrant ===")
    for i, emb in enumerate(embeddata.embeddings):
        print(f"\nProcessing page {i+1}/{len(embeddata.embeddings)}")
        points = [{
            "id": i,  # One ID per page
            "vector": emb.tolist(),  # All vectors for this page
            "payload": {"page": i}
        }]
        print(f"Created {len(points)} points for page {i+1}")
        
        for batch_start in range(0, len(points), 100):
            batch_end = min(batch_start + 100, len(points))
            batch = points[batch_start:batch_end]
            print(f"\nUploading batch {batch_start//100 + 1}")
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    print(f"Upload attempt {retry_count + 1}/{max_retries}")
                    client.upsert(collection_name="pdf_docs", points=batch)
                    success = True
                    print("Batch uploaded successfully")
                except Exception as e:
                    retry_count += 1
                    print(f"Error uploading batch: {e}")
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
            
            if not success:
                print(f"Failed to upload batch after {max_retries} attempts")

# Load OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Helper function to convert PIL image to base64
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")  # Save as JPEG
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def should_show_image(query):
    # Check if user wants to see the image
    show_image_phrases = [
        "show image",
        "display image",
        "show the image",
        "display the image",
        "see the image",
        "show figure",
        "display figure"    
    ]
    return any(phrase in query.lower() for phrase in show_image_phrases)

def extract_page_from_query(query):
    import re
    page_match = re.search(r'page\s+(\d+)', query.lower())
    if page_match:
        page_num = int(page_match.group(1)) - 1  # Convert to 0-based index
        return page_num
    return None

# Modify the save_image function to be more robust
def save_image(image, page_num):
    # Create an 'output' directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    output_path = f"output/page_{page_num+1}.jpg"
    image.save(output_path)
    print(f"\n>>> IMAGE SAVED: {output_path} <<<")
    return output_path

# Process the query with simple response handling
print("\n=== Processing Search Query ===")
query = "summerize this pdf for me, also expalins all the figures as well "
query_emb = embeddata.embed_query(query)

print("\n=== Preparing Search Vector ===")
if query_emb.ndim == 3:
    print("Using all token embeddings from 3D tensor for multivector search")
    all_vectors = query_emb[0].tolist()
    print(f"Using {len(all_vectors)} token embeddings for search")
elif query_emb.ndim == 2:
    print("Using all vectors from 2D tensor")
    all_vectors = query_emb.tolist()
else:
    print("Using raw vector")
    all_vectors = [query_emb.tolist()]
print(f"Total vectors: {len(all_vectors)}")

print("\n=== Searching Qdrant ===")
response = client.query_points(
    collection_name="pdf_docs",
    query=all_vectors,
    limit=5,
    with_payload=True
)

search_results = response.points
print(f"Found {len(search_results)} results")
if len(search_results) > 0:
    print("Top result scores:")
    for i, result in enumerate(search_results[:3]):
        print(f"Result {i+1}: score = {result.score}, page = {result.payload['page']}")

# Extract and save image if page is mentioned
page_num = extract_page_from_query(query)
if page_num is not None and should_show_image(query) and 0 <= page_num < len(images):
    image_path = save_image(images[page_num], page_num)
    print(f"\n>>> The image from page {page_num + 1} has been saved to: {image_path}")
    print("Please open this file to view the image.")

# Get GPT response (always provide text description)
response = client_openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": query},
            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_to_base64(page)}"}} 
              for page in images]
        ]}
    ]
)
print("\n=== Response ===")
print(response.choices[0].message.content)
