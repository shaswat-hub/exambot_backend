from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models
class AdminLogin(BaseModel):
    username: str
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    message: str

class AdBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position: str  # "left1", "left2", "right1", "right2", "top", "bottom"
    image_url: str
    link_url: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AdBlockUpdate(BaseModel):
    position: str
    image_url: str
    link_url: str

class ImageUploadRequest(BaseModel):
    images: List[str]  # base64 encoded images

class AIResponse(BaseModel):
    result: str

class VisitorStats(BaseModel):
    daily: int
    weekly: int
    monthly: int
    realtime: int

class VisitorLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str

# Routes
@api_router.get("/")
async def root():
    return {"message": "Exam Bot API is running"}

@api_router.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(credentials: AdminLogin):
    if credentials.username == "shaswat369" and credentials.password == "shaswat.millionaire":
        return AdminLoginResponse(success=True, message="Login successful")
    return AdminLoginResponse(success=False, message="Invalid credentials")

@api_router.get("/admin/ads", response_model=List[AdBlock])
async def get_ads():
    ads = await db.ad_blocks.find({}, {"_id": 0}).to_list(1000)
    for ad in ads:
        if isinstance(ad.get('updated_at'), str):
            ad['updated_at'] = datetime.fromisoformat(ad['updated_at'])
    return ads

@api_router.post("/admin/ads")
async def update_ad(ad_update: AdBlockUpdate):
    ad = AdBlock(**ad_update.model_dump())
    doc = ad.model_dump()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    # Upsert based on position
    await db.ad_blocks.update_one(
        {"position": ad.position},
        {"$set": doc},
        upsert=True
    )
    return {"success": True, "message": "Ad updated successfully"}

@api_router.post("/generate/summary", response_model=AIResponse)
async def generate_summary(request: ImageUploadRequest):
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        chat = LlmChat(
            api_key=api_key,
            session_id=str(uuid.uuid4()),
            system_message="You are a helpful study assistant. Analyze the provided images and create a comprehensive summary of the educational content."
        )
        chat.with_model("gemini", "gemini-3-flash-preview")
        
        # Prepare image contents
        image_contents = [ImageContent(image_base64=img) for img in request.images]
        
        user_message = UserMessage(
            text="Please analyze these images and provide a detailed summary of all the content. Include all important topics, concepts, and key points discussed in the images. Make it comprehensive and well-organized.",
            file_contents=image_contents
        )
        
        response = await chat.send_message(user_message)
        return AIResponse(result=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/generate/questions", response_model=AIResponse)
async def generate_questions(request: ImageUploadRequest):
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        chat = LlmChat(
            api_key=api_key,
            session_id=str(uuid.uuid4()),
            system_message="You are an expert exam question creator. Analyze the provided study material images and generate comprehensive exam questions."
        )
        chat.with_model("gemini", "gemini-3-flash-preview")
        
        # Prepare image contents
        image_contents = [ImageContent(image_base64=img) for img in request.images]
        
        user_message = UserMessage(
            text="Based on these study materials, generate a comprehensive question paper with maximum important questions that could appear in exams. Include various types of questions: multiple choice, short answer, and long answer questions. Focus on the most important and exam-relevant topics.",
            file_contents=image_contents
        )
        
        response = await chat.send_message(user_message)
        return AIResponse(result=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/visitor/track")
async def track_visitor(ip_address: str = "unknown"):
    visitor = VisitorLog(ip_address=ip_address)
    doc = visitor.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.visitor_logs.insert_one(doc)
    return {"success": True}

@api_router.get("/admin/stats", response_model=VisitorStats)
async def get_visitor_stats():
    now = datetime.now(timezone.utc)
    
    # Daily stats (last 24 hours)
    daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_count = await db.visitor_logs.count_documents({
        "timestamp": {"$gte": daily_start.isoformat()}
    })
    
    # Weekly stats (last 7 days)
    from datetime import timedelta
    weekly_start = (now - timedelta(days=7)).isoformat()
    weekly_count = await db.visitor_logs.count_documents({
        "timestamp": {"$gte": weekly_start}
    })
    
    # Monthly stats (last 30 days)
    monthly_start = (now - timedelta(days=30)).isoformat()
    monthly_count = await db.visitor_logs.count_documents({
        "timestamp": {"$gte": monthly_start}
    })
    
    # Realtime (last 5 minutes)
    realtime_start = (now - timedelta(minutes=5)).isoformat()
    realtime_count = await db.visitor_logs.count_documents({
        "timestamp": {"$gte": realtime_start}
    })
    
    return VisitorStats(
        daily=daily_count,
        weekly=weekly_count,
        monthly=monthly_count,
        realtime=realtime_count
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@app.on_event("shutdown")
async def shutdown_db_client():