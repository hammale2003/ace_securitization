"""
FastAPI Backend for the ACE Securitization System.

Provides REST API and SSE streaming endpoints for integration with
React/Next.js frontend or other clients.
"""
import json
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import PlaybookManager, Playbook, deduplicate_playbook
from agents import ACEPipeline
from trainer import TrainerPipeline, TrainerConfig
from trainer.granularity import GranularityLevel
from errors import ConfigurationError, LLMError, PlaybookError, ValidationError as ACEValidationError, log_error
from middleware import RateLimitMiddleware, ErrorHandlerMiddleware
from utils import logger


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
    mode: str = "answer"  # answer, enrich, derive, remediate, explore
    output_format: str = "text"  # text or prosemirror
    ground_truth: Optional[str] = None
    feedback: Optional[str] = None
    # Reformulation parameters
    reference_clause: Optional[str] = None
    constraints: Optional[str] = None
    issues: Optional[str] = None
    user_prompt: Optional[str] = None
    additional_instructions: Optional[str] = None
    full_pipeline: bool = True
    stream: bool = False


class GenerateRequest(BaseModel):
    """Request model for generation only."""
    question: str
    mode: str = "answer"  # answer, enrich, derive, remediate, explore
    output_format: str = "text"  # text or prosemirror
    # Reformulation parameters
    reference_clause: Optional[str] = None
    constraints: Optional[str] = None
    issues: Optional[str] = None
    user_prompt: Optional[str] = None
    additional_instructions: Optional[str] = None
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
    final_answer_prosemirror: Optional[dict] = None
    reformulation_result: Optional[dict] = None


class ReflectorResponse(BaseModel):
    """Response model for reflector output."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: list
    extracted_strategies: Optional[list] = None
    extracted_pitfalls: Optional[list] = None
    ground_truth_definition: Optional[str] = None


class CuratorResponse(BaseModel):
    """Response model for curator output."""
    reasoning: str
    operations: list


class PipelineResponse(BaseModel):
    """Response model for full pipeline execution."""
    question: str
    mode: str
    output_format: str
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
        self.trainer_pipeline: Optional[TrainerPipeline] = None
        self.config: Optional[ACEConfig] = None
        self.initialized: bool = False
    
    def initialize(self, config: ACEConfig):
        """Initialize the ACE pipeline."""
        self.config = config
        self.pipeline = ACEPipeline(config)
        
        # Initialize trainer pipeline
        trainer_config = TrainerConfig(llm_config=config.llm)
        self.trainer_pipeline = TrainerPipeline(
            config=trainer_config,
            playbook_manager=self.pipeline.playbook_manager,
            retriever=self.pipeline.retriever
        )
        
        self.initialized = True
    
    def reset(self):
        """Reset the application state."""
        self.pipeline = None
        self.trainer_pipeline = None
        self.config = None
        self.initialized = False


app_state = AppState()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("ACE Securitization API starting up...")
    yield
    logger.info("ACE Securitization API shutting down...")
    app_state.reset()


app = FastAPI(
    title="ACE Securitization API",
    description="Agentic Context Engineering for Securitization and Structured Finance",
    version="1.0.0",
    lifespan=lifespan
)

# Error handling middleware (must be first)
app.add_middleware(ErrorHandlerMiddleware)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, burst_size=10)

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
        # Validate provider
        valid_providers = ["openai", "anthropic", "google", "mock"]
        if request.llm_config.provider not in valid_providers:
            raise ACEValidationError(
                f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
                {"provider": request.llm_config.provider}
            )
        
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
        logger.info("Pipeline initialized successfully", extra={"provider": llm_config.provider, "model": llm_config.model})
        
        return {
            "status": "initialized",
            "config": {
                "provider": llm_config.provider,
                "model": llm_config.model,
                "playbook_path": playbook_config.path
            }
        }
        
    except (ACEValidationError, ConfigurationError) as e:
        log_error(logger, e, {"endpoint": "/initialize"})
        raise HTTPException(status_code=400, detail=e.message)
    except ValueError as e:
        log_error(logger, e, {"endpoint": "/initialize"})
        raise HTTPException(status_code=400, detail="Invalid configuration")
    except Exception as e:
        log_error(logger, e, {"endpoint": "/initialize"})
        raise HTTPException(status_code=500, detail="Failed to initialize pipeline")


@app.post("/generate", response_model=GeneratorResponse)
async def generate(request: GenerateRequest):
    """Generate an answer without reflection or curation."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise ACEValidationError("Question cannot be empty")
        
        if request.stream:
            return StreamingResponse(
                stream_generate(request),
                media_type="text/event-stream"
            )
        
        # Build kwargs based on mode
        kwargs = {
            "output_format": request.output_format
        }
        
        if request.mode != "answer":
            kwargs["mode"] = request.mode
            kwargs["additional_instructions"] = request.additional_instructions or ""
            
            if request.mode == "derive" and request.constraints:
                kwargs["constraints"] = request.constraints
            elif request.mode == "remediate" and request.issues:
                kwargs["issues"] = request.issues
            elif request.mode == "explore" and request.user_prompt:
                kwargs["user_prompt"] = request.user_prompt
        
        logger.info("Generating response", extra={"mode": request.mode, "output_format": request.output_format})
        output = app_state.pipeline.generate_only(request.question, **kwargs)
        
        return GeneratorResponse(
            reasoning=output.reasoning,
            bullet_ids=output.bullet_ids,
            final_answer=output.final_answer,
            final_answer_prosemirror=output.final_answer_prosemirror if hasattr(output, 'final_answer_prosemirror') else None,
            reformulation_result=output.reformulation_result if hasattr(output, 'reformulation_result') else None
        )
        
    except ACEValidationError as e:
        log_error(logger, e, {"endpoint": "/generate"})
        raise HTTPException(status_code=400, detail=e.message if hasattr(e, 'message') else str(e))
    except LLMError as e:
        log_error(logger, e, {"endpoint": "/generate"})
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    except Exception as e:
        log_error(logger, e, {"endpoint": "/generate"})
        raise HTTPException(status_code=500, detail="Failed to generate response")


async def stream_generate(request: GenerateRequest) -> AsyncGenerator[str, None]:
    """Stream generator output using SSE."""
    def stream_callback(chunk: str):
        pass  # Streaming handled by SSE
    
    loop = asyncio.get_event_loop()
    
    # Build kwargs based on mode
    kwargs = {
        "output_format": request.output_format,
        "stream_callback": stream_callback
    }
    
    if request.mode != "answer":
        kwargs["mode"] = request.mode
        kwargs["additional_instructions"] = request.additional_instructions or ""
        
        if request.mode == "derive" and request.constraints:
            kwargs["constraints"] = request.constraints
        elif request.mode == "remediate" and request.issues:
            kwargs["issues"] = request.issues
        elif request.mode == "explore" and request.user_prompt:
            kwargs["user_prompt"] = request.user_prompt
    
    output = await loop.run_in_executor(
        None,
        lambda: app_state.pipeline.generate_only(
            request.question,
            **kwargs
        )
    )
    
    yield f"data: {json.dumps({'type': 'complete', 'data': output.to_dict()})}\n\n"


@app.post("/run", response_model=PipelineResponse)
async def run_pipeline(request: QuestionRequest):
    """Run the full ACE pipeline."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise ACEValidationError("Question cannot be empty")
        
        if request.stream:
            return StreamingResponse(
                stream_pipeline(request),
                media_type="text/event-stream"
            )
        
        # Build kwargs based on mode
        kwargs = {
            "ground_truth": request.ground_truth,
            "feedback": request.feedback,
            "output_format": request.output_format
        }
        
        if request.mode != "answer":
            kwargs["mode"] = request.mode
            kwargs["additional_instructions"] = request.additional_instructions or ""
            
            if request.mode == "derive" and request.constraints:
                kwargs["constraints"] = request.constraints
            elif request.mode == "remediate" and request.issues:
                kwargs["issues"] = request.issues
            elif request.mode == "explore" and request.user_prompt:
                kwargs["user_prompt"] = request.user_prompt
        
        logger.info("Running full pipeline", extra={"mode": request.mode, "output_format": request.output_format})
        result = app_state.pipeline.run(
            question=request.question,
            **kwargs
        )
        
        return PipelineResponse(
            question=result.question,
            mode=request.mode,
            output_format=request.output_format,
            generator_output=GeneratorResponse(**result.generator_output.to_dict()),
            reflector_output=ReflectorResponse(**result.reflector_output.to_dict()) if result.reflector_output else None,
            curator_output=CuratorResponse(**result.curator_output.to_dict()) if result.curator_output else None,
            added_bullets=[b.to_dict() for b in result.added_bullets],
            playbook_stats=PlaybookStatsResponse(**{
                "total_bullets": result.playbook_stats.get("total_bullets", 0),
                "sections": result.playbook_stats.get("sections", {})
            }),
            timestamp=result.timestamp
        )
        
    except ACEValidationError as e:
        log_error(logger, e, {"endpoint": "/run"})
        raise HTTPException(status_code=400, detail=e.message if hasattr(e, 'message') else str(e))
    except (LLMError, PlaybookError) as e:
        log_error(logger, e, {"endpoint": "/run"})
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    except Exception as e:
        log_error(logger, e, {"endpoint": "/run"})
        raise HTTPException(status_code=500, detail="Failed to process request")


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
        
        # Build kwargs based on mode
        kwargs = {
            "ground_truth": request.ground_truth,
            "feedback": request.feedback,
            "output_format": request.output_format,
            "stream_callbacks": callbacks
        }
        
        if request.mode != "answer":
            kwargs["mode"] = request.mode
            kwargs["additional_instructions"] = request.additional_instructions or ""
            
            if request.mode == "derive" and request.constraints:
                kwargs["constraints"] = request.constraints
            elif request.mode == "remediate" and request.issues:
                kwargs["issues"] = request.issues
            elif request.mode == "explore" and request.user_prompt:
                kwargs["user_prompt"] = request.user_prompt
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: app_state.pipeline.run(
                question=request.question,
                **kwargs
            )
        )
        
        return result
    
    result = await run_with_streaming()
    
    yield f"data: {json.dumps({'type': 'generator', 'data': result.generator_output.to_dict()})}\n\n"
    if result.reflector_output:
        yield f"data: {json.dumps({'type': 'reflector', 'data': result.reflector_output.to_dict()})}\n\n"
    if result.curator_output:
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
# TRAINER MODE ENDPOINTS
# =============================================================================

class TrainerEnrichRequest(BaseModel):
    """Request model for trainer enrichment."""
    json_document: str  # Minified JSON document string
    extraction_types: Optional[List[str]] = None
    min_confidence: float = 0.5
    granularity_level: str = "batch"  # operative_clause_by_clause, batch, or full_document
    preview_only: bool = False


@app.post("/trainer/enrich")
async def trainer_enrich(request: TrainerEnrichRequest):
    """
    Enrich playbook from a securitization document.
    
    Accepts minified JSON documents (Master Framework Agreements, etc.)
    and extracts knowledge to enrich the playbook.
    """
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    if not app_state.trainer_pipeline:
        raise HTTPException(status_code=500, detail="Trainer pipeline not initialized")
    
    try:
        # Update config if provided
        if request.extraction_types:
            app_state.trainer_pipeline.config.extraction_types = request.extraction_types
        app_state.trainer_pipeline.config.min_extraction_confidence = request.min_confidence
        
        # Set granularity level
        try:
            app_state.trainer_pipeline.config.granularity_level = GranularityLevel(request.granularity_level)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid granularity_level. Must be one of: operative_clause_by_clause, batch, full_document"
            )
        
        # Parse document first
        document = app_state.trainer_pipeline.document_parser.parse_json_string(
            request.json_document
        )
        
        # Preview mode
        if request.preview_only:
            preview = app_state.trainer_pipeline.get_extraction_preview(document)
            return {
                "mode": "preview",
                "preview": preview
            }
        
        # Full enrichment
        result = app_state.trainer_pipeline.run_from_document(document)
        
        return {
            "mode": "enrich",
            "result": result.to_dict(),
            "summary": result.get_summary()
        }
    
    except json.JSONDecodeError as e:
        log_error(logger, e, {"endpoint": "/trainer/enrich"})
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except PlaybookError as e:
        log_error(logger, e, {"endpoint": "/trainer/enrich"})
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        log_error(logger, e, {"endpoint": "/trainer/enrich"})
        raise HTTPException(status_code=500, detail="Enrichment failed")


@app.post("/trainer/parse")
async def trainer_parse_document(request: TrainerEnrichRequest):
    """
    Parse a document and return its structure (no enrichment).
    
    Useful for previewing document structure before enrichment.
    """
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    if not app_state.trainer_pipeline:
        raise HTTPException(status_code=500, detail="Trainer pipeline not initialized")
    
    try:
        # Parse document
        document = app_state.trainer_pipeline.document_parser.parse_json_string(
            request.json_document
        )
        
        # Get hierarchy
        hierarchy = app_state.trainer_pipeline.document_parser.get_clause_hierarchy(document)
        
        return {
            "document_uid": document.document_uid,
            "document_type": document.document_type,
            "title": document.title,
            "total_clauses": len(document.get_all_clauses_flat()),
            "metadata": document.metadata,
            "hierarchy": hierarchy
        }
    
    except json.JSONDecodeError as e:
        log_error(logger, e, {"endpoint": "/trainer/parse"})
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        log_error(logger, e, {"endpoint": "/trainer/parse"})
        raise HTTPException(status_code=500, detail="Parsing failed")


@app.get("/trainer/stats")
async def get_trainer_stats():
    """Get statistics about trainer mode usage."""
    if not app_state.initialized:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    playbook = app_state.pipeline.get_playbook()
    stats = playbook.get_stats()
    
    return {
        "total_bullets": stats.get("total_bullets", 0),
        "sections": stats.get("sections", {}),
        "archived_count": stats.get("archived_count", 0)
    }


@app.get("/trainer/granularity-levels")
async def get_granularity_levels():
    """Get available granularity levels with descriptions."""
    return {
        "levels": [
            {
                "value": GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE.value,
                "name": "Operative Clause-by-Clause",
                "description": GranularityLevel.get_description(GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE),
                "cost": GranularityLevel.get_cost_indicator(GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE),
                "accuracy": GranularityLevel.get_accuracy_indicator(GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE)
            },
            {
                "value": GranularityLevel.BATCH.value,
                "name": "Batch Processing",
                "description": GranularityLevel.get_description(GranularityLevel.BATCH),
                "cost": GranularityLevel.get_cost_indicator(GranularityLevel.BATCH),
                "accuracy": GranularityLevel.get_accuracy_indicator(GranularityLevel.BATCH)
            },
            {
                "value": GranularityLevel.FULL_DOCUMENT.value,
                "name": "Full Document",
                "description": GranularityLevel.get_description(GranularityLevel.FULL_DOCUMENT),
                "cost": GranularityLevel.get_cost_indicator(GranularityLevel.FULL_DOCUMENT),
                "accuracy": GranularityLevel.get_accuracy_indicator(GranularityLevel.FULL_DOCUMENT)
            }
        ],
        "default": GranularityLevel.BATCH.value
    }

# =============================================================================
# RUN SERVER
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
