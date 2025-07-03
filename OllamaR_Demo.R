
# =============================================================================
# OLLAMAR TUTORIAL TOPICS
# =============================================================================

# 1. Basic setup and connection testing
# 2. Simple text generation
# 3. Interactive chat conversations
# 4. Code generation with specialized models
# 5. Code explanation and analysis
# 6. Creative writing with custom parameters
# 7. Structured data analysis
# 8. Multi-model comparison
# 9. Text embeddings and similarity
# 10. Batch processing
# 11. Advanced chat with system prompts
# 12. Error handling and robustness
# ==============================================================================

library(ollamar)

# =============================================================================
# 1. BASIC SETUP AND CONNECTION TEST
# =============================================================================

# Test connection to Ollama
test_connection()

# List available models
models <- list_models()
print(models)

# =============================================================================
# 2. SIMPLE TEXT GENERATION WITH LLAMA3.2
# =============================================================================

# Basic text generation
# Extract the actual response content using httr2
simple_response <- generate(
  model = "llama3.2:latest",
  prompt = "Explain quantum computing in simple terms for a 10-year-old"
)
response_content <- httr2::resp_body_json(simple_response)
cat("Simple Generation:\n", response_content$response, "\n\n")

# =============================================================================
# 3. INTERACTIVE CHAT CONVERSATION
# =============================================================================

# Start a chat conversation
chat_response1 <- chat(
  model = "llama3.2:latest",
  messages = list(
    list(role = "user", content = "I'm planning a trip to Norway. What are 3 must-see places?")
  )
)
chat_content <- httr2::resp_body_json(chat_response1)
cat("Text:\n", chat_content$message$content, "\n\n")



# =============================================================================
# 4. CODE GENERATION WITH CODELLAMA
# =============================================================================

# Generate Python code
python_code <- generate(
  model = "codellama:7b",
  prompt = "Write a Python function that calculates the following Financial metrics: NPV, IRR, and Payback Period for a given cash flow series"
)
response_content <- httr2::resp_body_json(python_code)
cat("Text:\n", response_content$response, "\n\n")

# Generate R code
r_code <- generate(
  model = "codellama:7b",
  prompt = "Write an R function that creates a beautiful ggplot2 visualization of the penguins dataset"
)
response_content <- httr2::resp_body_json(r_code)
cat("Text:\n", response_content$response, "\n\n")

# =============================================================================
# 5. CODE EXPLANATION WITH DEEPSEEK-CODER
# =============================================================================

# Explain complex code
code_explanation <- generate(
  model = "deepseek-coder:6.7b",
  prompt = "Explain this R code step by step:\n\nlibrary(dplyr)\niris %>% \n  group_by(Species) %>% \n  summarise(avg_length = mean(Sepal.Length), \n            count = n()) %>% \n  arrange(desc(avg_length))"
)
response_content <- httr2::resp_body_json(code_explanation)
cat("Text:\n", response_content$response, "\n\n")

# =============================================================================
# 6. CREATIVE WRITING WITH CUSTOM OPTIONS
# =============================================================================

# Creative story with custom parameters
# try first: 
#    temperature = 0.8 and 
#    top_p = 0.9
# Then:
#    temperature = 0.2 and
#    top_p = 0.1, 

creative_story <- generate(
  model = "llama3.2:latest",
  prompt = "Write a short story about the origin and the development of the double entry system of Accounting",
  temperature = 0.8,   # Higher creativity
  top_p = 0.9,         # More diverse responses
  num_predict = 300    # Limit response length
)
response_content <- httr2::resp_body_json(creative_story)
cat("Text:\n", response_content$response, "\n\n")

# =============================================================================
# 7. STRUCTURED DATA ANALYSIS REQUEST
# =============================================================================

# Ask for structured analysis
data_analysis <- generate(
  model = "llama3.2:latest",
  prompt = "Analyze the pros and cons of remote work vs office work. Present your answer in a structured format with clear headings."
)
response_content <- httr2::resp_body_json(data_analysis)
cat("Text:\n", response_content$response, "\n\n")

# =============================================================================
# 8. MULTIPLE MODEL COMPARISON
# =============================================================================

# Compare responses from different models on the same prompt
prompt_question <- "What are the key differences between machine learning and artificial intelligence?"

# Response from Llama3.2
llama_response <- generate(
  model = "llama3.2:latest",
  prompt = prompt_question
)

# Response from DeepSeek Coder
deepseek_response <- generate(
  model = "deepseek-coder:6.7b",
  prompt = prompt_question
)

cat("=== MODEL COMPARISON ===\n")
response_content <- httr2::resp_body_json(llama_response)
cat("\n=====================\nLLAMA3.2 Response::\n=====================\n", response_content$response, "\n\n")
response_content <- httr2::resp_body_json(llama_response)
cat("\n========================\nDEEPSEEK-CODER Response:\n========================\n", response_content$response, "\n\n")


# =============================================================================
# 9. EMBEDDINGS FOR TEXT SIMILARITY
# =============================================================================

# Generate embeddings for text similarity
text1 <- "I love R and data science"
text2 <- "I went to the zoo and looked at the wild animals."
text3 <- "I think that the Tiger is the most awsome animal in the world"
text4 <- "In Africa you will find wild lions as well as elephants."

# Get embeddings
embedding1 <- embed(model = "nomic-embed-text:latest", input = text1)
embedding2 <- embed(model = "nomic-embed-text:latest", input = text2)
embedding3 <- embed(model = "nomic-embed-text:latest", input = text3)
embedding4 <- embed(model = "nomic-embed-text:latest", input = text4)

# Calculate cosine similarity (basic implementation)
cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

sim_1_2 <- cosine_similarity(embedding1, embedding2)
sim_1_3 <- cosine_similarity(embedding1, embedding3)
sim_2_3 <- cosine_similarity(embedding2, embedding3)
sim_3_4 <- cosine_similarity(embedding3, embedding4)

cat("=== TEXT SIMILARITY ===\n")
cat("Similarity between text1 and text2:", round(sim_1_2, 3), "\n")
cat("Similarity between text1 and text3:", round(sim_1_3, 3), "\n")
cat("Similarity between text2 and text3:", round(sim_2_3, 3), "\n")
cat("Similarity between text3 and text4:", round(sim_3_4, 3), "\n")

# =============================================================================
# 10. BATCH PROCESSING EXAMPLE
# =============================================================================

# Process multiple prompts efficiently
topics <- c("Climate change", "Artificial Intelligence", "Space exploration")

batch_results <- list()
for(i in seq_along(topics)) {
  cat("Processing topic", i, "of", length(topics), ":", topics[i], "\n")
  
  result <- generate(
    model = "llama3.2:latest",
    prompt = paste("Explain", topics[i], "in exactly 2 sentences"),
    num_predict = 100
  )
  
  result_content <- httr2::resp_body_json(result)
  batch_results[[topics[i]]] <- result_content$response
}

cat("\n=== BATCH RESULTS ===\n")
for(topic in names(batch_results)) {
  cat(topic, ":\n", batch_results[[topic]], "\n\n")
}

# =============================================================================
# 11. ADVANCED CHAT WITH SYSTEM PROMPT
# =============================================================================

# Chat with system instructions
advanced_chat <- chat(
  model = "llama3.2:latest",
  messages = list(
    list(role = "system", content = "You are a helpful data science tutor. Always provide practical examples and suggest R code when relevant."),
    list(role = "user", content = "How do I handle missing values in my dataset?")
  )
)
response_content <- httr2::resp_body_json(advanced_chat)
response_content

cat("Model:\n", response_content$model, "\n\n")
cat("Created:\n", response_content$created_at, "\n\n")
cat("Created By:\n", response_content$message$role, "\n\n")
cat("Text:\n", response_content$message$content, "\n\n")


# =============================================================================
# 12. ERROR HANDLING AND ROBUST IMPLEMENTATION
# =============================================================================

# Robust function with error handling
safe_generate <- function(model, prompt, max_retries = 3) {
  for(i in 1:max_retries) {
    tryCatch({
      result <- generate(model = model, prompt = prompt)
      result_content <- httr2::resp_body_json(result)
      return(result_content$response)  # Return the extracted response text
    }, error = function(e) {
      cat("Attempt", i, "failed:", e$message, "\n")
      if(i == max_retries) {
        stop("Failed after", max_retries, "attempts")
      }
      Sys.sleep(1)  # Wait before retry
    })
  }
}

# Test the robust function
safe_result <- safe_generate(
  model = "llama3.2:latest",
  prompt = "What is the capital of Norway?"
)
cat("Safe Generation Result:\n", safe_result, "\n\n")

# ==========================  END OF SCRIPT  ==========================
