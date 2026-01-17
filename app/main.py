from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
import io
import uvicorn
from PIL import Image

from .config import config
from .database import db
from .watcher import watcher
from .tags import tag_processor
from .services.search_service import search_service, ImageEntry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    watcher.start()
    yield
    # Shutdown
    watcher.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom route for root app to handle mobile detection
@app.get("/app")
@app.get("/app/")
async def serve_app(request: Request):
    user_agent = request.headers.get("user-agent", "").lower()
    # Simple mobile detection
    if any(keyword in user_agent for keyword in ["android", "iphone", "ipad", "mobile"]):
        return FileResponse("web_local/index_mobile.html")
    return FileResponse("web_local/index.html")

# Static files (legacy support)
app.mount("/assets", StaticFiles(directory="web_local/assets"), name="static")
app.mount("/app", StaticFiles(directory="web_local", html=False), name="frontend")


@app.post("/API/search", response_model=List[ImageEntry])
async def search_images(
    q: Optional[str] = Form(None),
    image: List[UploadFile] = File(None),
    image_weight: Optional[List[float]] = Form(None),
    # Support "image_wight" for compatibility if sent
    image_wight: Optional[List[float]] = Form(None), 
    limit: int = Form(100)
):
    # Handle compatibility
    weights = image_weight if image_weight is not None else image_wight
    
    query_images = []
    if image:
        for upload_file in image:
            content = await upload_file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            query_images.append(img)
            
    results = search_service.search(
        query_text=q,
        query_images=query_images,
        image_weights=weights,
        limit=limit
    )
            
    return results

@app.get("/API/media/{entry_id}")
async def get_media(entry_id: int, size: Optional[str] = None):
    path = db.get_path_by_id(entry_id)
    if not path:
        raise HTTPException(status_code=404, detail="Not found")
    
    # If no size requested, return original file
    if not size:
        return FileResponse(path)

    # Parse Size
    try:
        w_str, h_str = size.lower().split('x')
        target_w = int(w_str)
        target_h = int(h_str)
    except ValueError:
        # Fallback to original if invalid format
        return FileResponse(path)

    # Process Thumbnail
    try:
        # Run image processing in thread pool to avoid blocking async loop
        def process_image():
            with Image.open(path) as img:
                img = img.convert("RGB")
                
                # Center Crop Logic
                img_w, img_h = img.size
                
                # Target aspect ratio
                target_aspect = target_w / target_h
                img_aspect = img_w / img_h
                
                if img_aspect > target_aspect:
                    # Image is wider than target: Crop width
                    new_w = int(img_h * target_aspect)
                    offset = (img_w - new_w) // 2
                    box = (offset, 0, offset + new_w, img_h)
                else:
                    # Image is taller than target: Crop height
                    new_h = int(img_w / target_aspect)
                    offset = (img_h - new_h) // 2
                    box = (0, offset, img_w, offset + new_h)
                
                img = img.crop(box)
                img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
                
                # Save to buffer
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                return buf

        # Run in threadpool
        import asyncio
        loop = asyncio.get_event_loop()
        buf = await loop.run_in_executor(None, process_image)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print(f"Error creating thumbnail for {path}: {e}")
        # Fallback
        return FileResponse(path)

@app.get("/API/tags")
async def get_tags(prefix: str = ""):
    if not prefix:
        return []
        
    df = tag_processor.df
    # filter
    matched = df[df["name"].str.startswith(prefix, na=False)]
    # sort by count
    # return list of (name, category, count)
    # limit 20
    matched = matched.head(20)
    return [(row["name"], row["category"], row["count"]) for _, row in matched.iterrows()]

from pydantic import BaseModel
class TagProbability(BaseModel):
    tag_name: str
    probability: float

@app.get("/API/tags_from_id/{entry_id}", response_model=List[TagProbability])
async def get_tags_from_id(entry_id: int, threshold: float = 0.1):
    # This logic combines DB and Feature Extractor, leaving here for now or could move to service.
    # Given the clear separation request, maybe move to service later or acceptable here as it's simple.
    # Let's keep it here but could be SearchService.get_tags_for_image(id)
    
    # Get embedding from DB
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM images WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
    if not row:
        raise HTTPException(404, "Image not found")
        
    embedding = row[0] # numpy array
    from .models import feature_extractor # Import here or global? Global available in imports
    import numpy as np
    
    probs = feature_extractor.get_tag_probabilities(embedding.reshape(1, -1))
    probs = probs[0] # (num_tags,)
    
    results = []
    indices = np.where(probs > threshold)[0]
    
    for idx in indices:
        tag_name = tag_processor.tag_names[idx]
        p = float(probs[idx])
        results.append(TagProbability(tag_name=tag_name, probability=p))
        
    results.sort(key=lambda x: x.probability, reverse=True)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
