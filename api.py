"""
FastAPI Backend for the ACE Securitization System.

Provides REST API and SSE streaming endpoints for integration with
React/Next.js frontend or other clients.
"""
import json
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import PlaybookManager, Playbook, deduplicate_playbook
from agents import ACEPipeline


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    api_key: Optional[str] = None


class InitializeRequest(BaseModel):
    """Request model for pipeline initialization."""
    llm_config: LLMConfigRequest = Field(default_factory=LLMConfigRequest)
    playbook_path: str = "playbook.json"
    max_reflector_iterations: int = 3


class QuestionRequest(BaseModel):
    """Request model for submitting a question."""
    question: str
    ground_truth: Optional[str] = None
    feedback: Optional[str] = None
    full_pipeline: bool = True
    stream: bool = False


class GenerateRequest(BaseModel):
    """Request model for generation only."""
    question: str
    stream: bool = False


class BulletResponse(BaseModel):
    """Response model for a playbook bullet."""
    id: str
    content: str
    helpful_count: int
    harmful_count: int
    neutral_count: int


class PlaybookStatsResponse(BaseModel):
    """Response model for playbook statistics."""
    total_bullets: int
    sections: Dict[str, int]


class GeneratorResponse(BaseModel):
    """Response model for generator output."""
    reasoning: str
    bullet_ids: list
    final_answer: str


class ReflectorResponse(BaseModel):
    """Response model for reflector output."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: list


class CuratorResponse(BaseModel):
    """Response model for curator output."""
    reasoning: str
    operations: list


class PipelineResponse(BaseModel):
    """Response model for full pipeline execution."""
    question: str
    generator_output: GeneratorResponse
    reflector_output: Optional[ReflectorResponse]
    curator_output: Optional[CuratorResponse]
    added_bullets: list
    playbook_stats: PlaybookStatsResponse
    timestamp: str


# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """Application state management."""
    
    def __init__(self):
        self.pipeline: Optional[ACEPipeline] = None
        self.config: Optional[ACEConfig] = None
        self.initialized: bool = False
    
    def initialize(self, config: ACEConfig):
        """Initialize the ACE pipeline."""
        self.config = config
        self.pipeline = ACEPipeline(config)
        self.initialized = True
    
    def reset(self):
        """Reset the application state."""
        self.pipeline = None
        self.config = None
        self.initialized = False


app_state = AppState()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    print("ACE Securitization API starting up...")
    yield
    print("ACE Securitization API shutting down...")
    app_state.reset()


app = FastAPI(
    title="ACE Securitization API",
    description="Agentic Context Engineering for Securitization and Structured Finance",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ACE Securitization API",
        "version": "1.0.0",
        "status": "running",
        "initialized": app_state.initialized
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": app_state.initialized,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/initialize")
async def initialize_pipeline(request: InitializeRequest):
    """Initialize the ACE pipeline with configuration."""
    try:
        llm_config = LLMConfig(
            provider=request.llm_config.provider,
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            api_key=request.llm_config.api_key,
            stream=True
        )
        
        playbook_config = PlaybookConfig(path=request.playbook_path)
        
        ace_config = ACEConfig(
            llm=llm_config,
            playbook=playbook_config,
            max_reflector_iterations=request.max_reflector_iterations
        )
        
        app_state.initialize(ace_config)
        
        return {
            "status": "initialized",
            "config": {
                "provider": llm_config.provider,
                "model": llm_config.model,
                "playbook_path": playbook_config.path
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GeneratorResponse)
async def generate(request: GenerateRequest):
    """Generate an answer without reflection or curation."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_generate(request.question),
                media_type="text/event-stream"
            )
        
        output = app_state.pipeline.generate_only(request.question)
        
        return GeneratorResponse(
            reasoning=output.reasoning,
            bullet_ids=output.bullet_ids,
            final_answer=output.final_answer
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generate(question: str) -> AsyncGenerator[str, None]:
    """Stream generator output using SSE."""
    collected_response = ""
    
    def stream_callback(chunk: str):
        nonlocal collected_response
        collected_response += chunk
    
    loop = asyncio.get_event_loop()
    
    output = await loop.run_in_executor(
        None,
        lambda: app_state.pipeline.generate_only(
            question,
            stream_callback=stream_callback
        )
    )
    
    yield f"data: {json.dumps({'type': 'complete', 'data': output.to_dict()})}\n\n"


@app.post("/run", response_model=PipelineResponse)
async def run_pipeline(request: QuestionRequest):
    """Run the full ACE pipeline."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_pipeline(request),
                media_type="text/event-stream"
            )
        
        result = app_state.pipeline.run(
            question=request.question,
            ground_truth=request.ground_truth,
            feedback=request.feedback
        )
        
        return PipelineResponse(
            question=result.question,
            generator_output=GeneratorResponse(**result.generator_output.to_dict()),
            reflector_output=ReflectorResponse(**result.reflector_output.to_dict()),
            curator_output=CuratorResponse(**result.curator_output.to_dict()),
            added_bullets=[b.to_dict() for b in result.added_bullets],
            playbook_stats=PlaybookStatsResponse(**{
                "total_bullets": result.playbook_stats.get("total_bullets", 0),
                "sections": result.playbook_stats.get("sections", {})
            }),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_pipeline(request: QuestionRequest) -> AsyncGenerator[str, None]:
    """Stream pipeline execution using SSE."""
    
    async def run_with_streaming():
        stages = {}
        
        def make_callback(stage: str):
            def callback(chunk: str):
                stages.setdefault(stage, "")
                stages[stage] += chunk
            return callback
        
        callbacks = {
            "generator": make_callback("generator"),
            "reflector": make_callback("reflector"),
            "curator": make_callback("curator")
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: app_state.pipeline.run(
                question=request.question,
                ground_truth=request.ground_truth,
                feedback=request.feedback,
                stream_callbacks=callbacks
            )
        )
        
        return result
    
    result = await run_with_streaming()
    
    yield f"data: {json.dumps({'type': 'generator', 'data': result.generator_output.to_dict()})}\n\n"
    yield f"data: {json.dumps({'type': 'reflector', 'data': result.reflector_output.to_dict()})}\n\n"
    yield f"data: {json.dumps({'type': 'curator', 'data': result.curator_output.to_dict()})}\n\n"
    yield f"data: {json.dumps({'type': 'complete', 'data': result.to_dict()})}\n\n"


@app.get("/playbook")
async def get_playbook():
    """Get the current playbook."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    playbook = app_state.pipeline.get_playbook()
    return playbook.to_dict()


@app.get("/playbook/stats", response_model=PlaybookStatsResponse)
async def get_playbook_stats():
    """Get playbook statistics."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    stats = app_state.pipeline.get_playbook_stats()
    return PlaybookStatsResponse(
        total_bullets=stats.get("total_bullets", 0),
        sections=stats.get("sections", {})
    )


@app.get("/playbook/{section}")
async def get_playbook_section(section: str):
    """Get bullets from a specific playbook section."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    playbook = app_state.pipeline.get_playbook()
    bullets = playbook.get_section(section)
    
    if bullets is None:
        raise HTTPException(status_code=404, detail=f"Section '{section}' not found")
    
    return {
        "section": section,
        "bullets": [b.to_dict() for b in bullets]
    }


@app.post("/playbook/deduplicate")
async def deduplicate_playbook_endpoint():
    """Deduplicate the playbook."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    playbook = app_state.pipeline.get_playbook()
    removed_ids = deduplicate_playbook(playbook)
    app_state.pipeline.playbook_manager.save()
    
    return {
        "removed_count": len(removed_ids),
        "removed_ids": removed_ids
    }


@app.post("/playbook/reset")
async def reset_playbook():
    """Reset the playbook to empty state."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    app_state.pipeline.playbook_manager._playbook = Playbook()
    app_state.pipeline.playbook_manager.save()
    
    return {"status": "reset", "message": "Playbook has been reset to empty state"}


@app.get("/bullet/{bullet_id}")
async def get_bullet(bullet_id: str):
    """Get a specific bullet by ID."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    playbook = app_state.pipeline.get_playbook()
    bullet = playbook.get_bullet_by_id(bullet_id)
    
    if bullet is None:
        raise HTTPException(status_code=404, detail=f"Bullet '{bullet_id}' not found")
    
    return bullet.to_dict()


# =============================================================================
# RUN SERVER
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
