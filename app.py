"""
Streamlit Web Interface for the ACE Securitization System.

Generator with integrated modes:
- answer: Standard Q&A using playbook
- enrich: Expand clause without changing meaning
- derive: Generate variants under constraints
- remediate: Fix compliance issues
- explore: Open-ended reformulation
"""
import streamlit as st
import json
import time
from typing import Optional

from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import deduplicate_playbook
from agents import ACEPipeline
from playbook_enricher import EnrichmentPipeline, EnrichmentConfig
from playbook_enricher.granularity import GranularityLevel


st.set_page_config(
    page_title="ACE Securitization",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "enrichment_pipeline" not in st.session_state:
        st.session_state.enrichment_pipeline = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "config" not in st.session_state:
        st.session_state.config = None
    if "output_format" not in st.session_state:
        st.session_state.output_format = "text"
    if "mode" not in st.session_state:
        st.session_state.mode = "answer"


init_session_state()


def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("Configuration")
    
    # LLM Settings
    st.sidebar.subheader("LLM Settings")
    
    provider = st.sidebar.selectbox(
        "Provider",
        ["openai", "anthropic", "google", "mock"],
        index=0,
        help="Select the LLM provider"
    )
    
    model_options = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "google": ["gemini-3-pro-preview","gemini-2.5-pro"],
        "mock": ["mock-model"]
    }
    
    model = st.sidebar.selectbox(
        "Model",
        model_options.get(provider, ["gpt-4"]),
        index=0
    )
    
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Leave blank to use environment variable"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    
    max_tokens = st.sidebar.number_input(
        "Max Tokens",
        min_value=1,
        max_value=32000,
        value=4096,
        step=256
    )
    
    enable_streaming = st.sidebar.checkbox(
        "Enable Streaming",
        value=True
    )
    
    st.sidebar.divider()
    
    # Output Format
    st.sidebar.subheader("Output Format")
    
    output_format = st.sidebar.radio(
        "Response Format",
        options=["text", "prosemirror"],
        index=0 if st.session_state.output_format == "text" else 1,
        help="text = plain text, prosemirror = JSON for editors"
    )
    st.session_state.output_format = output_format
    
    if output_format == "prosemirror":
        st.sidebar.info("ProseMirror JSON enabled")
    
    st.sidebar.divider()
    
    # Playbook Settings
    st.sidebar.subheader("Playbook Settings")
    
    playbook_path = st.sidebar.text_input(
        "Playbook Path",
        value="playbook.json"
    )
    
    # ACE Settings
    st.sidebar.subheader("ACE Settings")
    
    max_reflector_iterations = st.sidebar.number_input(
        "Max Reflector Iterations",
        min_value=1,
        max_value=10,
        value=3
    )
    
    st.sidebar.divider()
    
    # Display current configuration if pipeline is initialized
    if 'config' in st.session_state and st.session_state.config:
        st.sidebar.info(f"Modèle actif: **{st.session_state.config.llm.model}**\n\n Provider: **{st.session_state.config.llm.provider}**")
    
    # Initialize Pipeline
    if st.sidebar.button(" Initialize Pipeline", type="primary", use_container_width=True):
        try:
            llm_config = LLMConfig(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=enable_streaming,
                api_key=api_key if api_key else None
            )
            
            playbook_config = PlaybookConfig(path=playbook_path)
            
            ace_config = ACEConfig(
                llm=llm_config,
                playbook=playbook_config,
                max_reflector_iterations=max_reflector_iterations,
                enable_streaming=enable_streaming
            )
            
            st.session_state.pipeline = ACEPipeline(ace_config)
            st.session_state.config = ace_config
            
            # Initialize enrichment pipeline
            enrichment_config = EnrichmentConfig(llm_config=llm_config)
            st.session_state.enrichment_pipeline = EnrichmentPipeline(
                config=enrichment_config,
                playbook_manager=st.session_state.pipeline.playbook_manager,
                retriever=st.session_state.pipeline.retriever
            )
            
            st.sidebar.success("Pipeline initialized!")
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    # Status
    if st.session_state.pipeline:
        st.sidebar.success("Pipeline: Active")
        stats = st.session_state.pipeline.get_playbook_stats()
        st.sidebar.metric("Total Bullets", stats.get("total_bullets", 0))
        
        mode_labels = {
            "answer": "Q&A",
            "enrich": "Enrich",
            "derive": "Derive",
            "remediate": "Remediate",
            "explore": "Explore"
        }
        mode_label = mode_labels.get(st.session_state.mode, st.session_state.mode)
        st.sidebar.caption(f"Mode: **{mode_label}**")
        st.sidebar.caption(f"Output: **{st.session_state.output_format}**")
    else:
        st.sidebar.warning("Pipeline: Not initialized")


def render_header():
    """Render main header."""
    st.title("ACE Securitization System")
    st.caption("Advanced Clause Engine with Multi-Mode Generation")


def render_input():
    """Render unified input section with mode selector."""
    st.subheader("Generator")
    
    # Mode selector with descriptions
    mode_options = {
        "answer": "**Answer** - Standard Q&A using playbook knowledge",
        "enrich": "**Enrich** - Expand clause without changing legal meaning",
        "derive": "**Derive** - Generate variants under specific constraints",
        "remediate": "**Remediate** - Fix compliance issues and restore alignment",
        "explore": "**Explore** - Open-ended reformulation with custom instructions"
    }
    
    mode = st.radio(
        "Select Mode:",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        horizontal=True,
        key="mode_radio"
    )
    st.session_state.mode = mode
    
    st.divider()
    
    # Common inputs
    if mode == "answer":
        # Q&A mode
        question = st.text_area(
            "Question:",
            height=120,
            placeholder="e.g., What are the essential elements of a true sale opinion?",
            key="input_main"
        )
        
        # Training options
        with st.expander("Training Options (Optional)"):
            ground_truth = st.text_area(
                "Expected Answer:",
                height=100,
                help="Provide the correct answer to improve Reflector analysis",
                key="input_ground_truth"
            )
            
            feedback = st.text_area(
                "Additional Feedback:",
                height=50,
                key="input_feedback"
            )
        
        # Reformulation params (empty for answer mode)
        reference_clause = ""
        constraints = ""
        issues = ""
        user_prompt = ""
        additional_instructions = ""
        
    else:
        # Reformulation modes
        question = st.text_area(
            "Clause to Reformulate:",
            height=150,
            placeholder="Paste the clause you want to reformulate...",
            key="input_main"
        )
        
        reference_clause = ""  # Removed from UI
        
        # Mode-specific inputs
        if mode == "derive":
            st.markdown("**Constraints:**")
            constraints = st.text_area(
                "Specify constraints/rules:",
                height=80,
                placeholder="e.g., 'Must include 5-day grace period' or '[if all issuer accounts are in the EU] [if accounts span multiple jurisdictions]'",
                key="input_constraints"
            )
            issues = ""
            user_prompt = ""
            
        elif mode == "remediate":
            st.markdown("**Issues to Fix:**")
            issues = st.text_area(
                "Identify problems:",
                height=80,
                placeholder="e.g., 'Ambiguous language undermines true sale', 'Missing materiality threshold', 'Inconsistent defined terms'",
                key="input_issues"
            )
            constraints = ""
            user_prompt = ""
            
        elif mode == "explore":
            st.markdown("**Your Request:**")
            user_prompt = st.text_area(
                "What would you like to do?",
                height=80,
                placeholder="e.g., 'Make this simpler', 'Add fallback provisions', 'Soften the tone', 'Draft UK-style version'",
                key="input_user_prompt"
            )
            constraints = ""
            issues = ""
            
        else:  # enrich
            constraints = ""
            issues = ""
            user_prompt = ""
        
        # Additional instructions for all reformulation modes
        with st.expander("Additional Instructions (Optional)"):
            additional_instructions = st.text_area(
                "Extra guidance:",
                height=60,
                placeholder="Any additional context or requirements...",
                key="input_additional"
            )
        
        # Training options for reformulation modes
        with st.expander("Training Options (Optional)"):
            ground_truth = st.text_area(
                "Expected Reformulation:",
                height=100,
                help="Provide the correct/ideal reformulation to improve Reflector analysis",
                key="input_ground_truth"
            )
            
            feedback = st.text_area(
                "Additional Feedback:",
                height=50,
                help="Any specific guidance for this reformulation",
                key="input_feedback"
            )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_full = st.button(
            "Run Full Pipeline",
            type="primary",
            use_container_width=True,
            help="Run Generator → Reflector → Curator"
        )
    
    with col2:
        generate_only = st.button(
            "Generate Only",
            use_container_width=True,
            help="Run Generator only (skip Reflector & Curator)"
        )
    
    return {
        "question": question,
        "mode": mode,
        "ground_truth": ground_truth,
        "feedback": feedback,
        "reference_clause": reference_clause,
        "constraints": constraints,
        "issues": issues,
        "user_prompt": user_prompt,
        "additional_instructions": additional_instructions or "",
        "run_full": run_full,
        "generate_only": generate_only
    }


def run_pipeline(inputs: dict, full_pipeline: bool):
    """Execute the ACE pipeline."""
    if not st.session_state.pipeline:
        st.error("Please initialize the pipeline first!")
        return
    
    if not inputs["question"].strip():
        st.warning("Please enter a question or clause.")
        return
    
    output_format = st.session_state.output_format
    
    enable_streaming = (
        st.session_state.config 
        and st.session_state.config.enable_streaming 
        and st.session_state.config.llm.stream
    )
    
    try:
        if full_pipeline:
            # Full pipeline with Reflector and Curator
            if enable_streaming:
                generator_placeholder = st.empty()
                reflector_placeholder = st.empty()
                curator_placeholder = st.empty()
                
                generator_text = ""
                reflector_text = ""
                curator_text = ""
                
                def generator_callback(chunk: str):
                    nonlocal generator_text
                    generator_text += chunk
                    generator_placeholder.markdown(f"**Generator:**\n\n{generator_text}")
                
                def reflector_callback(chunk: str):
                    nonlocal reflector_text
                    reflector_text += chunk
                    reflector_placeholder.markdown(f"**Reflector:**\n\n{reflector_text}")
                
                def curator_callback(chunk: str):
                    nonlocal curator_text
                    curator_text += chunk
                    curator_placeholder.markdown(f"**Curator:**\n\n{curator_text}")
                
                stream_callbacks = {
                    "generator": generator_callback,
                    "reflector": reflector_callback,
                    "curator": curator_callback
                }
                
                result = st.session_state.pipeline.run(
                    question=inputs["question"],
                    ground_truth=inputs["ground_truth"] if inputs["ground_truth"] else None,
                    feedback=inputs["feedback"] if inputs["feedback"] else None,
                    stream_callbacks=stream_callbacks,
                    output_format=output_format,
                    mode=inputs["mode"],
                    reference_clause=inputs["reference_clause"],
                    constraints=inputs["constraints"],
                    issues=inputs["issues"],
                    user_prompt=inputs["user_prompt"],
                    additional_instructions=inputs["additional_instructions"]
                )
                
                generator_placeholder.empty()
                reflector_placeholder.empty()
                curator_placeholder.empty()
            else:
                with st.spinner("⏳ Processing..."):
                    result = st.session_state.pipeline.run(
                        question=inputs["question"],
                        ground_truth=inputs["ground_truth"] if inputs["ground_truth"] else None,
                        feedback=inputs["feedback"] if inputs["feedback"] else None,
                        output_format=output_format,
                        mode=inputs["mode"],
                        reference_clause=inputs["reference_clause"],
                        constraints=inputs["constraints"],
                        issues=inputs["issues"],
                        user_prompt=inputs["user_prompt"],
                        additional_instructions=inputs["additional_instructions"]
                    )
            
            st.session_state.current_result = {
                "generator_output": result.generator_output.to_dict(),
                "reflector_output": result.reflector_output.to_dict(),
                "curator_output": result.curator_output.to_dict(),
                "added_bullets": [b.to_dict() for b in result.added_bullets],
                "playbook_stats": result.playbook_stats,
                "mode": inputs["mode"]
            }
            
            st.session_state.history.append({
                "question": f"[{inputs['mode'].upper()}] {inputs['question'][:50]}...",
                "answer": result.generator_output.final_answer[:100] + "...",
                "output_format": output_format,
                "timestamp": result.timestamp
            })
            
        else:
            # Generate only
            if enable_streaming:
                generator_placeholder = st.empty()
                generator_text = ""
                
                def generator_callback(chunk: str):
                    nonlocal generator_text
                    generator_text += chunk
                    generator_placeholder.markdown(f"**Generator:**\n\n{generator_text}")
                
                output = st.session_state.pipeline.generate_only(
                    inputs["question"],
                    stream_callback=generator_callback,
                    output_format=output_format,
                    mode=inputs["mode"],
                    reference_clause=inputs["reference_clause"],
                    constraints=inputs["constraints"],
                    issues=inputs["issues"],
                    user_prompt=inputs["user_prompt"],
                    additional_instructions=inputs["additional_instructions"]
                )
                
                generator_placeholder.empty()
            else:
                with st.spinner("⏳ Generating..."):
                    output = st.session_state.pipeline.generate_only(
                        inputs["question"],
                        output_format=output_format,
                        mode=inputs["mode"],
                        reference_clause=inputs["reference_clause"],
                        constraints=inputs["constraints"],
                        issues=inputs["issues"],
                        user_prompt=inputs["user_prompt"],
                        additional_instructions=inputs["additional_instructions"]
                    )
            
            st.session_state.current_result = {
                "generator_output": output.to_dict(),
                "reflector_output": None,
                "curator_output": None,
                "added_bullets": [],
                "mode": inputs["mode"]
            }
            
            st.session_state.history.append({
                "question": f"[{inputs['mode'].upper()}] {inputs['question'][:50]}...",
                "answer": output.final_answer[:100] + "...",
                "output_format": output_format,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        st.success("Processing complete!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("View Traceback"):
            st.code(traceback.format_exc())


def render_results():
    """Render results section."""
    if st.session_state.current_result is None:
        st.info("Submit a question or clause to see results here.")
        return
    
    result = st.session_state.current_result
    mode = result.get("mode", "answer")
    
    # Mode badge
    st.caption(f"**Mode: {mode.upper()}**")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Generator",
        "Reflector",
        "Curator",
        "Playbook"
    ])
    
    with tab1:
        render_generator_output(result.get("generator_output"))
    
    with tab2:
        render_reflector_output(result.get("reflector_output"))
    
    with tab3:
        render_curator_output(result.get("curator_output"), result.get("added_bullets", []))
    
    with tab4:
        render_playbook_view()


def render_generator_output(output: Optional[dict]):
    """Render Generator output."""
    st.subheader("Generator Output")
    
    if not output:
        st.warning("No output available.")
        return
    
    # Check if this is a reformulation output
    reformulation_result = output.get("reformulation_result")
    
    if reformulation_result:
        # Reformulation mode - show structured alternatives
        if reformulation_result.get("success"):
            alternatives = reformulation_result.get("alternatives", [])
            st.success(f"Generated {len(alternatives)} alternative(s)")
            
            for alt in alternatives:
                rank = alt.get("rank", "?")
                confidence = alt.get("confidence", 0)
                content = alt.get("content", "")
                changes = alt.get("changes_summary", "")
                
                # Confidence indicator
                if confidence >= 0.8:
                    conf_indicator = "[HIGH]"
                elif confidence >= 0.6:
                    conf_indicator = "[MED]"
                else:
                    conf_indicator = "[LOW]"
                
                with st.expander(f"Alternative {rank} {conf_indicator} (Confidence: {confidence:.0%})", expanded=(rank == 1)):
                    # Content display
                    if isinstance(content, dict):
                        # ProseMirror format
                        st.markdown("**Clause:**")
                        
                        # Check if structured format with sub_clauses
                        if "main_clause" in content and "sub_clauses" in content:
                            # Structured format - render nicely
                            main_text = _extract_from_prosemirror_node(content["main_clause"])
                            st.markdown(main_text)
                            
                            # Render sub-clauses
                            for i, sub_clause in enumerate(content.get("sub_clauses", []), 1):
                                sub_text = _extract_from_prosemirror_node(sub_clause)
                                st.markdown(f"  {sub_text}")
                            
                            extracted_text = _extract_text_from_prosemirror(content)
                        else:
                            # Simple ProseMirror document
                            extracted_text = _extract_text_from_prosemirror(content)
                            st.markdown(extracted_text)
                        
                        with st.expander("ProseMirror JSON"):
                            st.json(content)
                            st.download_button(
                                "Download JSON",
                                json.dumps(content, indent=2),
                                file_name=f"reformulated_alt{rank}.json",
                                mime="application/json",
                                key=f"download_alt_{rank}"
                            )
                    else:
                        # Plain text
                        st.markdown("**Clause:**")
                        st.markdown(content)
                        extracted_text = content
                    
                    # Changes summary
                    if changes:
                        st.markdown(f"**Changes:** {changes}")
                    
                    # Copy text area
                    st.text_area(
                        "Copy text:",
                        value=content if isinstance(content, str) else extracted_text,
                        height=80,
                        key=f"copy_alt_{rank}",
                        label_visibility="collapsed"
                    )
        else:
            st.error(f"Reformulation failed: {reformulation_result.get('failure_reason', 'Unknown error')}")
        
        # Reasoning
        with st.expander("Reasoning", expanded=False):
            st.markdown(output.get("reasoning", "No reasoning provided."))
    
    else:
        # Standard Q&A mode
        st.markdown("### Final Answer")
        st.markdown(output.get("final_answer", "No answer generated."))
        
        # ProseMirror JSON (if available)
        prosemirror_output = output.get("final_answer_prosemirror")
        if prosemirror_output:
            with st.expander("ProseMirror JSON", expanded=False):
                st.json(prosemirror_output)
                st.download_button(
                    "Download JSON",
                    json.dumps(prosemirror_output, indent=2),
                    file_name="answer_prosemirror.json",
                    mime="application/json"
                )
        
        # Reasoning
        with st.expander("Reasoning", expanded=True):
            st.markdown(output.get("reasoning", "No reasoning provided."))
        
        # Bullet IDs
        bullet_ids = output.get("bullet_ids", [])
        if bullet_ids:
            st.markdown("### Playbook Bullets Used")
            for bid in bullet_ids:
                st.code(bid, language=None)
        else:
            st.info("No playbook bullets were referenced.")


def _extract_text_from_prosemirror(doc: dict) -> str:
    """Extract plain text from ProseMirror document."""
    if not doc or not isinstance(doc, dict):
        return ""
    
    # Check if this is structured content with main_clause and sub_clauses
    if "main_clause" in doc and "sub_clauses" in doc:
        texts = []
        
        # Extract from main clause (no body_doc wrapper)
        main_text = _extract_from_prosemirror_node(doc["main_clause"])
        if main_text:
            texts.append(main_text)
        
        # Extract from sub_clauses (no body_doc wrapper)
        for sub_clause in doc.get("sub_clauses", []):
            sub_text = _extract_from_prosemirror_node(sub_clause)
            if sub_text:
                texts.append(sub_text)
        
        return "\n".join(texts)
    else:
        # Simple ProseMirror document
        return _extract_from_prosemirror_node(doc)


def _extract_from_prosemirror_node(node: dict) -> str:
    """Extract text from a ProseMirror node (recursive helper)."""
    if not node or not isinstance(node, dict):
        return ""
    
    texts = []
    
    def extract(n):
        if isinstance(n, dict):
            if n.get("type") == "text":
                texts.append(n.get("text", ""))
            elif n.get("type") == "slot":
                # Skip slot placeholders
                pass
            elif "content" in n:
                for child in n["content"]:
                    extract(child)
        elif isinstance(n, list):
            for item in n:
                extract(item)
    
    extract(node)
    return " ".join(texts)


def render_reflector_output(output: Optional[dict]):
    """Render Reflector output."""
    st.subheader("Reflector Analysis")
    
    if not output:
        st.warning("No reflector output available.")
        return
    
    # Key Insight
    st.markdown("### Key Insight")
    st.info(output.get("key_insight", "No key insight extracted."))
    
    # Extracted Strategies (from ground truth comparison)
    extracted_strategies = output.get("extracted_strategies", [])
    if extracted_strategies:
        st.markdown("### Extracted Strategies")
        st.success("Strategies learned from ground truth comparison:")
        for i, strategy in enumerate(extracted_strategies, 1):
            st.markdown(f"{i}. {strategy}")
    
    # Extracted Pitfalls (from ground truth comparison)
    extracted_pitfalls = output.get("extracted_pitfalls", [])
    if extracted_pitfalls:
        st.markdown("### Extracted Pitfalls")
        st.warning("Pitfalls identified from ground truth comparison:")
        for i, pitfall in enumerate(extracted_pitfalls, 1):
            st.markdown(f"{i}. {pitfall}")
    
    # Error Identification
    with st.expander("Error Identification", expanded=True):
        st.markdown(output.get("error_identification", "No errors identified."))
    
    # Root Cause Analysis
    with st.expander("Root Cause Analysis"):
        st.markdown(output.get("root_cause_analysis", "No root cause analysis provided."))
    
    # Correct Approach
    with st.expander("Correct Approach"):
        st.markdown(output.get("correct_approach", "No correct approach suggested."))
    
    # Full Reasoning
    with st.expander("Full Reasoning"):
        st.markdown(output.get("reasoning", "No reasoning provided."))
    
    # Bullet Tags
    bullet_tags = output.get("bullet_tags", [])
    if bullet_tags:
        st.markdown("### Bullet Tags")
        for tag in bullet_tags:
            tag_indicators = {
                "helpful": "[+]",
                "harmful": "[-]",
                "neutral": "[~]"
            }
            indicator = tag_indicators.get(tag.get("tag"), "[~]")
            st.markdown(f"{indicator} `{tag.get('id')}`: **{tag.get('tag')}**")


def render_curator_output(output: Optional[dict], added_bullets: list):
    """Render Curator output."""
    st.subheader("Curator Updates")
    
    if not output:
        st.warning("No curator output available.")
        return
    
    # Reasoning
    with st.expander("Curator's Reasoning", expanded=True):
        st.markdown(output.get("reasoning", "No reasoning provided."))
    
    # Operations
    operations = output.get("operations", [])
    
    if operations:
        st.markdown("### Planned Operations")
        for i, op in enumerate(operations, 1):
            op_type = op.get('type', 'UNKNOWN')
            op_labels = {
                "ADD": "[+]",
                "REMOVE": "[-]",
                "MODIFY": "[*]",
                "MERGE": "[M]"
            }
            label = op_labels.get(op_type, "[?]")
            
            with st.expander(f"{label} Operation {i}: **{op_type}**"):
                if op_type == "ADD":
                    st.markdown(f"**Section:** {op.get('section')}")
                    st.markdown(f"**Content:** {op.get('content')}")
                elif op_type == "REMOVE":
                    st.markdown(f"**Bullet ID:** `{op.get('bullet_id')}`")
                    st.markdown(f"**Reason:** {op.get('reason')}")
                elif op_type == "MODIFY":
                    st.markdown(f"**Bullet ID:** `{op.get('bullet_id')}`")
                    st.markdown(f"**New Content:** {op.get('new_content')}")
                    st.markdown(f"**Reason:** {op.get('reason')}")
                elif op_type == "MERGE":
                    st.markdown(f"**Source Bullets:** {op.get('source_bullet_ids')}")
                    st.markdown(f"**Target Section:** {op.get('target_section')}")
                    st.markdown(f"**Merged Content:** {op.get('merged_content')}")
                    st.markdown(f"**Reason:** {op.get('reason')}")
    else:
        st.info("No new operations needed - playbook already contains relevant knowledge.")
    
    # Added Bullets
    if added_bullets:
        st.markdown("### Added Bullets")
        for bullet in added_bullets:
            st.success(f"**{bullet.get('id')}**: {bullet.get('content')[:100]}...")


def render_playbook_view():
    """Render playbook viewer."""
    st.subheader("Current Playbook")
    
    if not st.session_state.pipeline:
        st.warning("Initialize the pipeline to view the playbook.")
        return
    
    playbook = st.session_state.pipeline.get_playbook()
    stats = playbook.get_stats()
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", stats.get("total_bullets", 0))
    col2.metric("Strategies", stats.get("sections", {}).get("strategies", 0))
    col3.metric("Pitfalls", stats.get("sections", {}).get("pitfalls", 0))
    
    # Sections
    sections = ["strategies", "pitfalls", "definitions"]
    
    for section in sections:
        bullets = playbook.get_section(section)
        if bullets:
            
            with st.expander(f"**{section.upper()}** ({len(bullets)} items)", expanded=False):
                for bullet in bullets:
                    effectiveness = bullet.effectiveness_score
                    if effectiveness > 0.5:
                        eff_indicator = "[+]"
                    elif effectiveness >= 0:
                        eff_indicator = "[~]"
                    else:
                        eff_indicator = "[-]"
                    
                    st.markdown(f"""
                    **{bullet.id}** {eff_indicator}
                    
                    {bullet.content}
                    
                    *Helpful: {bullet.helpful_count} | Harmful: {bullet.harmful_count} | Neutral: {bullet.neutral_count}*
                    
                    ---
                    """)
    
    # Archived bullets
    if playbook.archived_bullets:
        with st.expander(f"**ARCHIVED** ({len(playbook.archived_bullets)} items)", expanded=False):
            for bullet in playbook.archived_bullets:
                st.markdown(f"""
                **{bullet.id}** *(archived)*
                
                {bullet.content}
                
                *Reason: {bullet.archive_reason}*
                
                ---
                """)
    
    # Actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Deduplicate", use_container_width=True):
            removed = deduplicate_playbook(playbook)
            if removed:
                st.session_state.pipeline.playbook_manager.save()
                st.success(f"Removed {len(removed)} duplicates")
            else:
                st.info("No duplicates found")
    
    with col2:
        if st.button("Auto-Cleanup", use_container_width=True):
            removed = st.session_state.pipeline.auto_cleanup()
            if removed:
                st.success(f"Removed {len(removed)} harmful bullets")
            else:
                st.info("No harmful bullets to remove")
    
    with col3:
        if st.button("Export", use_container_width=True):
            playbook_json = json.dumps(playbook.to_dict(), indent=2)
            st.download_button(
                "Download JSON",
                playbook_json,
                file_name="playbook_export.json",
                mime="application/json",
                use_container_width=True
            )


def render_history():
    """Render history."""
    st.subheader("📜 History")
    
    if not st.session_state.history:
        st.info("No questions asked yet.")
        return
    
    for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
        with st.expander(f"{i}. {item['question'][:50]}...", expanded=False):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer'][:200]}...")
            st.markdown(f"**Format:** {item.get('output_format', 'text')}")
            st.markdown(f"*{item['timestamp']}*")


def _estimate_processing_time(extraction_types: list, batch_size: int, max_clauses: int) -> str:
    """Estimate processing time based on settings."""
    num_types = len(extraction_types)
    num_batches = (max_clauses + batch_size - 1) // batch_size
    total_llm_calls = num_types * num_batches
    
    # Estimate ~3 seconds per LLM call
    estimated_seconds = total_llm_calls * 3
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds / 60
        return f"~{minutes:.1f} minutes"
    else:
        hours = estimated_seconds / 3600
        return f"~{hours:.1f} hours"


def render_enrichment_mode():
    """Render Enrichment Mode interface."""
    st.title("Playbook Enrichment")
    st.caption("Enrich playbook from real securitization documents using ACE framework")
    
    if not st.session_state.enrichment_pipeline:
        st.warning("Initialize the pipeline first to use Enrichment Mode")
        return
    
    
    st.divider()
    
    # Document input
    st.subheader("1. Upload Document")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        options=["Upload File", "Paste Text"],
        horizontal=True,
        help="Upload a file or paste JSON text directly"
    )
    
    json_input = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload JSON or TXT file:",
            type=["json", "txt"],
            help="Upload your minified JSON document from securitization transaction"
        )
        
        if uploaded_file:
            json_input = uploaded_file.read().decode("utf-8")
            st.success(f"✅ Loaded: {uploaded_file.name} ({len(json_input)} characters)")
            with st.expander("Preview uploaded content"):
                st.code(json_input[:500] + "..." if len(json_input) > 500 else json_input, language="json")
    
    else:  # Paste Text
        json_input = st.text_area(
            "Paste your JSON document here:",
            height=300,
            placeholder='{"document_uid": "...", "title": "...", "clauses": [...]}',
            help="Paste the complete JSON document text"
        )
        
        if json_input:
            st.success(f"✅ Text pasted ({len(json_input)} characters)")
            with st.expander("Preview pasted content"):
                st.code(json_input[:500] + "..." if len(json_input) > 500 else json_input, language="json")
    
    # Extraction settings
    st.subheader("2. Configure Extraction")
    
    # Section Selection
    st.markdown("**Target Sections:**")
    st.caption("Select which types of knowledge to extract from the document")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extract_strategies = st.checkbox("Strategies", value=True, help="Best practices and methodologies")
    with col2:
        extract_pitfalls = st.checkbox("Pitfalls", value=True, help="Common mistakes to avoid")
    with col3:
        extract_definitions = st.checkbox("Definitions", value=True, help="Key legal terms")
    
    # Build extraction sections list
    extraction_sections = []
    if extract_strategies:
        extraction_sections.append("strategies")
    if extract_pitfalls:
        extraction_sections.append("pitfalls")
    if extract_definitions:
        extraction_sections.append("definitions")
    
    if not extraction_sections:
        st.warning("Please select at least one section to extract")
    else:
        st.success(f"✓ Extracting: {', '.join(extraction_sections)}")
    
    st.markdown("---")
    
    # Granularity Level Selection
    st.markdown("**Granularity Level:**")
    
    granularity_level = st.selectbox(
        "Processing Granularity:",
        options=[
            GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE.value,
            GranularityLevel.FULL_DOCUMENT.value
        ],
        index=0,  # Default to OPERATIVE_CLAUSE_BY_CLAUSE
        format_func=lambda x: {
            "operative_clause_by_clause": "Operative Clause-by-Clause",
            "full_document": "Full Document"
        }.get(x, x),
        help="Clause-by-clause provides best quality with ACE framework evaluation per item."
    )
    
    granularity_enum = GranularityLevel(granularity_level)
    
    if granularity_enum == GranularityLevel.OPERATIVE_CLAUSE_BY_CLAUSE:
        st.info("Clause-by-clause mode: Each extracted item is evaluated by ACE for maximum quality.")
    else:
        st.info("Full document mode: Faster processing with ACE evaluation of extracted items.")
    
    st.markdown("---")
    
    # Batch Processing Settings
    st.markdown("**Batch Processing:**")
    
    max_items_per_batch = st.number_input(
        "Max Items Per Batch",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        help="Number of extracted knowledge items to process together in a single LLM call. Set to 1 for individual processing (most expensive). Higher values reduce costs but increase prompt size."
    )
    
    if max_items_per_batch > 1:
        st.info(f"✓ Batch processing enabled: {max_items_per_batch} items per batch. This will significantly reduce LLM calls and costs.")
    else:
        st.warning("⚠ Individual processing: Each item processed separately (most expensive but highest quality per item).")
    
    st.markdown("---")
    
    
    # Action buttons
    st.subheader("3. Run Extraction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preview_btn = st.button("Preview", type="secondary", use_container_width=True)
    
    with col2:
        enrich_btn = st.button("Enrich Playbook", type="primary", use_container_width=True)
    
    with col3:
        parse_btn = st.button("Parse Only", type="secondary", use_container_width=True)
    
    st.divider()
    
    # Results area
    if parse_btn and json_input:
        with st.spinner("Parsing document..."):
            try:
                document = st.session_state.enrichment_pipeline.document_parser.parse_json_string(json_input)
                
                st.success("Document parsed successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Document", document.title)
                col2.metric("Type", document.document_type)
                col3.metric("Total Clauses", len(document.get_all_clauses_flat()))
                
                with st.expander("Document Structure", expanded=True):
                    st.code(f"Document: {document.title}\nType: {document.document_type}\nTotal Clauses: {len(document.get_all_clauses_flat())}\nOperative Clauses: {len(document.get_operative_clauses())}", language="text")
            
            except Exception as e:
                st.error(f"Error parsing document: {str(e)}")
    
    if preview_btn and json_input:
        if not extraction_sections:
            st.error("Please select at least one section to extract")
        else:
            with st.spinner("Generating preview..."):
                try:
                    # Update config with selected sections, granularity, and batch size
                    st.session_state.enrichment_pipeline.config.extraction_sections = extraction_sections
                    st.session_state.enrichment_pipeline.config.granularity_level = granularity_enum
                    st.session_state.enrichment_pipeline.config.validator_batch_size = max_items_per_batch
                    
                    document = st.session_state.enrichment_pipeline.document_parser.parse_json_string(json_input)
                    
                    # Check document size and warn if large
                    num_operative_clauses = len(document.get_operative_clauses())
                    if granularity_enum == GranularityLevel.FULL_DOCUMENT and num_operative_clauses > 500:
                        st.info(f"ℹ️ Large document ({num_operative_clauses} clauses). Preview will show items from the first chunk.")
                    
                    preview = st.session_state.enrichment_pipeline.get_extraction_preview(document, max_items=20)
                
                    st.success("Preview generated!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Document", preview["document_title"])
                    col2.metric("Total Extracted", preview["total_extracted"])
                    col3.metric("Sections", ", ".join(preview.get("extraction_sections", [])))
                    
                    st.markdown("### Preview Items")
                    
                    for i, item in enumerate(preview["preview_items"], 1):
                        section_badge = item.get("section", "unknown")
                        with st.expander(f"**Item {i}** - [{section_badge}]", expanded=(i <= 3)):
                            st.info(item["content"])
                            st.caption(f"Source: {item['source']}")
                
                except Exception as e:
                    st.error(f"Error generating preview: {str(e)}")
    
    if enrich_btn and json_input:
        if not extraction_sections:
            st.error("Please select at least one section to extract")
        else:
            try:
                # Update config with selected sections, granularity, and batch size
                st.session_state.enrichment_pipeline.config.extraction_sections = extraction_sections
                st.session_state.enrichment_pipeline.config.granularity_level = granularity_enum
                st.session_state.enrichment_pipeline.config.validator_batch_size = max_items_per_batch
                
                document = st.session_state.enrichment_pipeline.document_parser.parse_json_string(json_input)
                
                # Check document size and warn if large
                num_operative_clauses = len(document.get_operative_clauses())
                if granularity_enum == GranularityLevel.FULL_DOCUMENT and num_operative_clauses > 500:
                    st.warning(f"⚠️ Large document detected ({num_operative_clauses} clauses). The document will be processed in chunks to avoid token limits. This may take several minutes.")
                
                # Run enrichment with progress
                with st.spinner(f"Enriching playbook using ACE framework ({granularity_level})..."):
                    result = st.session_state.enrichment_pipeline.run_from_document(document)
            
                st.success("Playbook enriched successfully with ACE framework!")
                
                # Show summary
                st.markdown(result.get_summary())
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Extracted", result.total_extracted)
                col2.metric("Processed", result.total_processed)
                col3.metric("Added", result.total_added)
                col4.metric("Skipped", result.total_skipped)
                
                # Added bullets
                if result.added_bullet_ids:
                    st.markdown("### Added Bullets")
                    playbook = st.session_state.pipeline.playbook_manager.get_playbook()
                    for bullet_id in result.added_bullet_ids[:10]:
                        bullet = playbook.get_bullet_by_id(bullet_id)
                        if bullet:
                            with st.expander(f"**{bullet_id}**"):
                                st.markdown(bullet.content)
                    
                    if len(result.added_bullet_ids) > 10:
                        st.caption(f"... and {len(result.added_bullet_ids) - 10} more bullets")
            
            except Exception as e:
                st.error(f"Error enriching playbook: {str(e)}")
                st.exception(e)


def main():
    """Main application entry point."""
    render_sidebar()
    
    # Page selector
    page = st.sidebar.radio(
        "Navigation:",
        options=["Generator", "Playbook Enrichment"],
        label_visibility="collapsed"
    )
    
    if page == "Generator":
        render_header()
        
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            inputs = render_input()
            
            if inputs["run_full"] or inputs["generate_only"]:
                run_pipeline(inputs, inputs["run_full"])
            
            st.divider()
            render_results()
        
        with col2:
            render_history()
    
    elif page == "Playbook Enrichment":
        render_enrichment_mode()


if __name__ == "__main__":
    main()