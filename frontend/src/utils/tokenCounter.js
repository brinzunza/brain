/**
 * Approximate token counting for the frontend
 * Based on OpenAI's rough estimate: 1 token ≈ 4 characters or 0.75 words
 * This provides a good enough estimate for real-time display
 */

/**
 * Estimate token count from text
 * Uses a combination of word count and character count for better accuracy
 * @param {string} text - The text to count tokens for
 * @returns {number} Estimated token count
 */
export function estimateTokenCount(text) {
  if (!text || text.trim().length === 0) {
    return 0
  }

  // Remove extra whitespace
  const cleaned = text.trim()

  // Method 1: Character-based (1 token ≈ 4 chars)
  const charCount = cleaned.length
  const charBasedTokens = charCount / 4

  // Method 2: Word-based (1 token ≈ 0.75 words)
  const words = cleaned.split(/\s+/).filter(word => word.length > 0)
  const wordBasedTokens = words.length / 0.75

  // Use average of both methods for better accuracy
  const estimated = Math.round((charBasedTokens + wordBasedTokens) / 2)

  return estimated
}

/**
 * Format token count for display
 * @param {number} tokens - Token count
 * @returns {string} Formatted string
 */
export function formatTokenCount(tokens) {
  if (tokens === 0) return '0 tokens'
  if (tokens === 1) return '1 token'
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(1)}k tokens`
  }
  return `${tokens} tokens`
}
