use adk_agent::LlmAgentBuilder;
use adk_model::ollama::{OllamaConfig, OllamaModel};
use adk_runner::{Runner, RunnerConfig};
use adk_session::{CreateRequest, InMemorySessionService, SessionService};
use adk_core::Content;
use futures_util::StreamExt;
use std::sync::Arc;
use anyhow::Result;

pub async fn send_to_ollama(
    instruction: &str,
    input: &str,
    limit_token: bool,
    num_predict: &str,
) -> Result<String> {
    send_to_ollama_with_context(instruction, input, limit_token, num_predict).await
}

pub async fn send_to_ollama_with_context(
    instruction: &str,
    input: &str,
    limit_token: bool,
    num_predict: &str,
) -> Result<String> {
    // Create Ollama model configuration
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "glm-4.7-flash:latest".to_string());
    let mut config = OllamaConfig::new(&model_name);
    
    // Set num_ctx if limit_token is enabled
    if limit_token {
        if let Ok(num) = num_predict.parse::<u32>() {
            config.num_ctx = Some(num);
        }
    }
    
    let model = OllamaModel::new(config)?;

    // Create agent with custom instruction
    let agent = LlmAgentBuilder::new("local-assistant")
        .description("A helpful local assistant")
        .model(Arc::new(model))
        .instruction(instruction)
        .build()?;

    // Create session service
    let session_service = Arc::new(InMemorySessionService::new());
    
    // Create session
    let session = session_service
        .create(CreateRequest {
            app_name: "web-agents".to_string(),
            user_id: "user1".to_string(),
            session_id: None,
            state: std::collections::HashMap::new(),
        })
        .await?;

    // Create runner
    let runner = Runner::new(RunnerConfig {
        app_name: "web-agents".to_string(),
        agent: Arc::new(agent),
        session_service,
        artifact_service: None,
        memory_service: None,
        plugin_manager: None,
        run_config: None,
        compaction_config: None,
    })?;

    // Create content with input text
    let user_content = Content::new("user").with_text(input);
    
    // Extract and print the input prompt (truncated for conversation context)
    let input_text = user_content.parts.iter()
        .find_map(|p| match p {
            adk_core::Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("");
    
    // Print truncated input for conversation context (safe UTF-8 truncation)
    if !input_text.is_empty() {
        // Truncate to approximately 200 bytes, but ensure we don't split UTF-8 characters
        let max_bytes = 200;
        if input_text.len() > max_bytes {
            // Find the last valid character boundary before max_bytes
            let mut truncate_at = max_bytes;
            while !input_text.is_char_boundary(truncate_at) && truncate_at > 0 {
                truncate_at -= 1;
            }
            println!("Context: {}...", &input_text[..truncate_at]);
        } else {
            println!("Context: {}", input_text);
        }
    }
    
    // Send prompt to Ollama
    let mut stream = runner
        .run("user1".to_string(), session.id().to_string(), user_content)
        .await?;
    
    // Collect the stream events and print tokens as they come
    let mut response_parts = Vec::new();
    
    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => {
                // Extract text from llm_response.content.parts
                if let Some(content) = event.llm_response.content.as_ref() {
                    for part in &content.parts {
                        match part {
                            adk_core::Part::Text { text } => {
                                // Print token as it arrives (streaming)
                                print!("{}", text);
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                                response_parts.push(text.clone());
                            }
                            _ => {
                                // Handle other part types if needed
                            }
                        }
                    }
                }
                
                // Check if this is the final event (turn_complete = true)
                if event.llm_response.turn_complete {
                    if let Some(usage) = &event.llm_response.usage_metadata {
                        println!("\n[Tokens: prompt={}, candidates={}, total={}]", 
                            usage.prompt_token_count, 
                            usage.candidates_token_count, 
                            usage.total_token_count);
                    }
                }
            }
            Err(e) => {
                println!("\n[Stream error: {}]", e);
                return Err(anyhow::anyhow!("Stream error: {}", e));
            }
        }
    }
    
    println!();
    Ok(response_parts.join(""))
}

pub async fn test_ollama() -> Result<String> {
    // Create Ollama model configuration
    // Assumes Ollama is running on localhost:11434
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "glm-4.7-flash:latest".to_string());
    let config = OllamaConfig::new(&model_name);
    let model = OllamaModel::new(config)?;

    // Create agent
    let agent = LlmAgentBuilder::new("local-assistant")
        .description("A helpful local assistant")
        .model(Arc::new(model))
        .instruction("You are a helpful assistant running locally via Ollama.")
        .build()?;

    // Create session service
    let session_service = Arc::new(InMemorySessionService::new());
    
    // Create session
    let session = session_service
        .create(CreateRequest {
            app_name: "web-agents".to_string(),
            user_id: "user1".to_string(),
            session_id: None,
            state: std::collections::HashMap::new(),
        })
        .await?;

    // Create runner
    let runner = Runner::new(RunnerConfig {
        app_name: "web-agents".to_string(),
        agent: Arc::new(agent),
        session_service,
        artifact_service: None,
        memory_service: None,
        plugin_manager: None,
        run_config: None,
        compaction_config: None,
    })?;

    // Create content with text
    let user_content = Content::new("user").with_text("Hello, how are you?");
    
    // Extract and print the input prompt
    let input_text = user_content.parts.iter()
        .find_map(|p| match p {
            adk_core::Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("");
    println!("Input: {}", input_text);
    
    // Send a simple test prompt - run takes user_id, session_id, and content
    let mut stream = runner
        .run("user1".to_string(), session.id().to_string(), user_content)
        .await?;
    
    // Collect the stream events and print tokens as they come
    let mut response_parts = Vec::new();
    print!("Response: ");
    
    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => {
                // Extract text from llm_response.content.parts
                // event.llm_response is a direct field, not an Option
                if let Some(content) = event.llm_response.content.as_ref() {
                    for part in &content.parts {
                        match part {
                            adk_core::Part::Text { text } => {
                                // Print token as it arrives (streaming)
                                print!("{}", text);
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                                response_parts.push(text.clone());
                            }
                            _ => {
                                // Handle other part types if needed
                            }
                        }
                    }
                }
                
                // Check if this is the final event (turn_complete = true)
                if event.llm_response.turn_complete {
                    if let Some(usage) = &event.llm_response.usage_metadata {
                        println!("\n\n[Tokens: prompt={}, candidates={}, total={}]", 
                            usage.prompt_token_count, 
                            usage.candidates_token_count, 
                            usage.total_token_count);
                    }
                }
            }
            Err(e) => {
                println!("\n[Stream error: {}]", e);
                return Err(anyhow::anyhow!("Stream error: {}", e));
            }
        }
    }
    
    println!();
    Ok(response_parts.join(""))
}

