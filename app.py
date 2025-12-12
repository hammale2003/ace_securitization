"""
Streamlit Web Interface for the ACE Securitization System.

Provides a user-friendly interface for:
- Asking securitization questions
- Viewing Generator, Reflector, and Curator outputs
- Exploring and managing the playbook
"""
import streamlit as st
import json
import time
from typing import Optional
from pathlib import Path

# Import ACE components
from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import PlaybookManager, Playbook, deduplicate_playbook
from agents import ACEPipeline, GeneratorOutput, ReflectorOutput, CuratorOutput


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ACE Securitization",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "config" not in st.session_state:
        st.session_state.config = None


init_session_state()


# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================

def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("Configuration")
    
    # LLM Provider Selection
    st.sidebar.subheader("LLM Settings")
    
    provider = st.sidebar.selectbox(
        "Provider",
        ["openai", "anthropic", "google", "mock"],
        index=0,
        help="Select the LLM provider"
    )
    
    # Model selection based on provider
    model_options = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "google": ["gemini-pro", "gemini-1.5-pro"],
        "mock": ["mock-model"]
    }
    
    model = st.sidebar.selectbox(
        "Model",
        model_options.get(provider, ["gpt-4"]),
        index=0
    )
    
    # API Key input
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter your API key (leave blank to use environment variable)"
    )
    
    # Temperature
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )
    
    # Max Tokens
    max_tokens = st.sidebar.number_input(
        "Max Tokens",
        min_value=1,
        max_value=32000,
        value=4096,
        step=256,
        help="Maximum number of tokens in the response"
    )
    
    # Enable streaming
    enable_streaming = st.sidebar.checkbox(
        "Enable Streaming",
        value=True,
        help="Stream tokens in real-time (provider must support it)"
    )
    
    st.sidebar.divider()
    
    # Playbook Settings
    st.sidebar.subheader("Playbook Settings")
    
    playbook_path = st.sidebar.text_input(
        "Playbook Path",
        value="playbook.json",
        help="Path to the playbook JSON file"
    )
    
    # ACE Settings
    st.sidebar.subheader("ACE Settings")
    
    max_reflector_iterations = st.sidebar.number_input(
        "Max Reflector Iterations",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of refinement iterations for the Reflector"
    )
    
    # Initialize Pipeline Button
    st.sidebar.divider()
    
    if st.sidebar.button("Initialize Pipeline", type="primary", use_container_width=True):
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
            st.sidebar.success("Pipeline initialized!")
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    # Show pipeline status
    if st.session_state.pipeline:
        st.sidebar.success("Pipeline: Active")
        stats = st.session_state.pipeline.get_playbook_stats()
        st.sidebar.metric("Total Bullets", stats.get("total_bullets", 0))
    else:
        st.sidebar.warning("Pipeline: Not initialized")


# =============================================================================
# MAIN CONTENT
# =============================================================================

def render_header():
    """Render the main header."""
    st.title("📜 ACE Securitization System")

def render_question_input():
    """Render the question input section."""
    st.subheader("Ask a Question")
    

    
    question = st.text_area(
        "Enter your securitization question:",
        value="",
        height=100,
        placeholder="e.g., What are the essential elements of a true sale opinion?"
    )
    
    # Optional ground truth for training
    ground_truth = None
    feedback = None
    
    with st.expander("Training Options (Optional)"):
        ground_truth = st.text_area(
            "Ground Truth / Expected Answer:",
            height=100,
            help="If you know the correct answer, enter it here to improve the Reflector's analysis."
        )
        
        feedback = st.text_area(
            "Additional Feedback:",
            height=50,
            help="Any specific feedback or guidance for this question."
        )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_full = st.button("Run Full Pipeline", type="primary", use_container_width=True)
    
    with col2:
        generate_only = st.button(" Generate Only", use_container_width=True)
    
    return question, ground_truth, feedback, run_full, generate_only


def render_results():
    """Render the results section."""
    if st.session_state.current_result is None:
        st.info("Submit a question to see results here.")
        return
    
    result = st.session_state.current_result
    
    # Create tabs for different outputs
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
    """Render the Generator's output."""
    st.subheader("Generator Output")
    
    if not output:
        st.warning("No generator output available.")
        return
    
    # Final Answer
    st.markdown("### Final Answer")
    st.markdown(output.get("final_answer", "No answer generated."))
    
    # Reasoning
    with st.expander("Reasoning", expanded=True):
        st.markdown(output.get("reasoning", "No reasoning provided."))
    
    # Bullet IDs used
    bullet_ids = output.get("bullet_ids", [])
    if bullet_ids:
        st.markdown("### Playbook Bullets Used")
        for bid in bullet_ids:
            st.code(bid)
    else:
        st.info("No playbook bullets were referenced.")


def render_reflector_output(output: Optional[dict]):
    """Render the Reflector's output."""
    st.subheader("Reflector Analysis")
    
    if not output:
        st.warning("No reflector output available.")
        return
    
    # Key Insight
    st.markdown("### Key Insight")
    st.info(output.get("key_insight", "No key insight extracted."))
    
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
        st.markdown("### 🏷️ Bullet Tags")
        for tag in bullet_tags:
            tag_color = {"helpful": "🟢", "harmful": "🔴", "neutral": "⚪"}.get(tag.get("tag"), "⚪")
            st.markdown(f"{tag_color} `{tag.get('id')}`: {tag.get('tag')}")


def render_curator_output(output: Optional[dict], added_bullets: list):
    """Render the Curator's output."""
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
        st.markdown("###  Planned Operations")
        for i, op in enumerate(operations, 1):
            with st.expander(f"Operation {i}: {op.get('type')} → {op.get('section')}"):
                st.markdown(f"**Section:** {op.get('section')}")
                st.markdown(f"**Content:** {op.get('content')}")
    else:
        st.info("No new operations needed - playbook already contains relevant knowledge.")
    
    # Added Bullets
    if added_bullets:
        st.markdown("###  Added Bullets")
        for bullet in added_bullets:
            st.success(f"**{bullet.get('id')}**: {bullet.get('content')[:100]}...")


def render_playbook_view():
    """Render the playbook viewer."""
    st.subheader("Current Playbook")
    
    if not st.session_state.pipeline:
        st.warning("Initialize the pipeline to view the playbook.")
        return
    
    playbook = st.session_state.pipeline.get_playbook()
    stats = playbook.get_stats()
    
    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bullets", stats.get("total_bullets", 0))
    col2.metric("Strategies", stats.get("sections", {}).get("strategies", 0))
    col3.metric("Pitfalls", stats.get("sections", {}).get("pitfalls", 0))
    col4.metric("Templates", stats.get("sections", {}).get("templates", 0))
    
    # Section views
    sections = ["strategies", "pitfalls", "templates", "definitions", "code_snippets"]
    
    for section in sections:
        bullets = playbook.get_section(section)
        if bullets:
            with st.expander(f"📁 {section.upper()} ({len(bullets)} items)", expanded=False):
                for bullet in bullets:
                    effectiveness = bullet.effectiveness_score
                    eff_indicator = "🟢" if effectiveness > 0.5 else "🟡" if effectiveness >= 0 else "🔴"
                    
                    st.markdown(f"""
                    **{bullet.id}** {eff_indicator}
                    
                    {bullet.content}
                    
                    *Helpful: {bullet.helpful_count} | Harmful: {bullet.harmful_count} | Neutral: {bullet.neutral_count}*
                    
                    ---
                    """)
    
    # Playbook actions
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Deduplicate Playbook"):
            removed = deduplicate_playbook(playbook)
            if removed:
                st.session_state.pipeline.playbook_manager.save()
                st.success(f"Removed {len(removed)} duplicate bullets.")
            else:
                st.info("No duplicates found.")
    
    with col2:
        if st.button(" Export Playbook"):
            playbook_json = json.dumps(playbook.to_dict(), indent=2)
            st.download_button(
                "Download JSON",
                playbook_json,
                file_name="playbook_export.json",
                mime="application/json"
            )


def render_history():
    """Render the question history."""
    st.subheader("📜 History")
    
    if not st.session_state.history:
        st.info("No questions asked yet.")
        return
    
    for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
        with st.expander(f"{i}. {item['question'][:50]}...", expanded=False):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer'][:200]}...")
            st.markdown(f"*{item['timestamp']}*")


# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def run_pipeline(question: str, ground_truth: str, feedback: str, full_pipeline: bool):
    """Execute the ACE pipeline."""
    if not st.session_state.pipeline:
        st.error("Please initialize the pipeline first!")
        return
    
    if not question.strip():
        st.warning("Please enter a question.")
        return
    
    # Check if streaming is enabled (both in ACEConfig and LLMConfig)
    enable_streaming = (
        st.session_state.config 
        and st.session_state.config.enable_streaming 
        and st.session_state.config.llm.stream
    )
    
    try:
        if full_pipeline:
            if enable_streaming:
                # Create streaming callbacks for each agent
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
                    question=question,
                    ground_truth=ground_truth if ground_truth else None,
                    feedback=feedback if feedback else None,
                    stream_callbacks=stream_callbacks
                )
                
                # Clear streaming placeholders after completion
                generator_placeholder.empty()
                reflector_placeholder.empty()
                curator_placeholder.empty()
            else:
                # Non-streaming mode
                with st.spinner("Processing..."):
                    result = st.session_state.pipeline.run(
                        question=question,
                        ground_truth=ground_truth if ground_truth else None,
                        feedback=feedback if feedback else None
                    )
                
            # Store results (same for both streaming and non-streaming)
            st.session_state.current_result = {
                "generator_output": result.generator_output.to_dict(),
                "reflector_output": result.reflector_output.to_dict(),
                "curator_output": result.curator_output.to_dict(),
                "added_bullets": [b.to_dict() for b in result.added_bullets],
                "playbook_stats": result.playbook_stats
            }
            
            # Add to history
            st.session_state.history.append({
                "question": question,
                "answer": result.generator_output.final_answer,
                "timestamp": result.timestamp
            })
            
        else:
            # Generate only
            if enable_streaming:
                # Streaming for generate_only
                generator_placeholder = st.empty()
                generator_text = ""
                
                def generator_callback(chunk: str):
                    nonlocal generator_text
                    generator_text += chunk
                    generator_placeholder.markdown(f"**Generator:**\n\n{generator_text}")
                
                output = st.session_state.pipeline.generate_only(
                    question,
                    stream_callback=generator_callback
                )
                
                generator_placeholder.empty()
            else:
                # Non-streaming mode
                with st.spinner("Generating..."):
                    output = st.session_state.pipeline.generate_only(question)
            
            st.session_state.current_result = {
                "generator_output": output.to_dict(),
                "reflector_output": None,
                "curator_output": None,
                "added_bullets": []
            }
            
            st.session_state.history.append({
                "question": question,
                "answer": output.final_answer,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        st.success("✅ Processing complete!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    render_header()
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question, ground_truth, feedback, run_full, generate_only = render_question_input()
        
        if run_full or generate_only:
            run_pipeline(question, ground_truth, feedback, run_full)
        
        st.divider()
        render_results()
    
    with col2:
        render_history()


if __name__ == "__main__":
    main()
