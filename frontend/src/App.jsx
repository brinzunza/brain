import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import './App.css'
import { estimateTokenCount } from './utils/tokenCounter'

const API_URL = 'http://localhost:8000'

function App() {
  const [textInput, setTextInput] = useState('')
  const [file, setFile] = useState(null)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [textStatus, setTextStatus] = useState('')
  const [fileStatus, setFileStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [clearStatus, setClearStatus] = useState('')
  const [llmProvider, setLlmProvider] = useState('chatgpt')
  const [availableProviders, setAvailableProviders] = useState({
    openai_available: true,
    ollama_available: false
  })
  const messagesEndRef = useRef(null)

  useEffect(() => {
    checkAvailableProviders()
  }, [])

  useEffect(() => {
    // Auto-scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const checkAvailableProviders = async () => {
    try {
      const response = await fetch(`${API_URL}/api/providers`)
      const data = await response.json()
      if (data.status === 'success') {
        setAvailableProviders(data.providers)

        // auto-select LLM provider based on availability
        if (!data.providers.openai_available && data.providers.ollama_available) {
          setLlmProvider('ollama')
        } else if (data.providers.openai_available) {
          setLlmProvider('chatgpt')
        }
      }
    } catch (err) {
      console.error('failed to check providers:', err)
    }
  }

  const handleTextSubmit = async () => {
    if (!textInput.trim()) return

    setTextStatus('storing...')
    try {
      const response = await fetch(`${API_URL}/api/add-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textInput.toLowerCase()
        })
      })

      const data = await response.json()
      if (data.status === 'success') {
        const tokenInfo = data.token_count ? ` (${data.token_count} tokens)` : ''
        setTextStatus(`stored${tokenInfo}`)
        setTextInput('')
        setTimeout(() => setTextStatus(''), 3000)
      } else {
        setTextStatus('error: ' + data.message)
      }
    } catch (error) {
      setTextStatus('error: ' + error.message)
    }
  }

  const handleFileSubmit = async () => {
    if (!file) return

    setFileStatus('storing...')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/api/add-file`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()
      if (data.status === 'success') {
        const tokenInfo = data.token_count ? ` (${data.token_count} tokens)` : ''
        setFileStatus(`stored${tokenInfo}`)
        setFile(null)
        setTimeout(() => setFileStatus(''), 3000)
      } else {
        setFileStatus('error: ' + data.message)
      }
    } catch (error) {
      setFileStatus('error: ' + error.message)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) return

    const userQuestion = question
    setQuestion('')
    setMessages(prev => [...prev, { type: 'question', content: userQuestion }])
    setLoading(true)

    // Add placeholder message for streaming
    const answerIndex = messages.length + 1
    setMessages(prev => [...prev, {
      type: 'answer',
      content: '',
      streaming: true,
      tokens: null
    }])

    try {
      const response = await fetch(`${API_URL}/api/ask-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuestion,
          llm_provider: llmProvider
        })
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let metadata = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'metadata') {
                metadata = data
              } else if (data.type === 'content') {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[answerIndex] = {
                    ...updated[answerIndex],
                    content: updated[answerIndex].content + data.content
                  }
                  return updated
                })
              } else if (data.type === 'done') {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[answerIndex] = {
                    ...updated[answerIndex],
                    streaming: false,
                    tokens: {
                      query: metadata?.query_tokens || 0,
                      context: metadata?.context_tokens || 0,
                      answer: data.answer_tokens,
                      total: data.total_tokens
                    }
                  }
                  return updated
                })
              } else if (data.type === 'error') {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[answerIndex] = {
                    type: 'error',
                    content: 'error: ' + data.message,
                    streaming: false
                  }
                  return updated
                })
              }
            } catch (e) {
              console.error('Error parsing stream data:', e)
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => {
        const updated = [...prev]
        updated[answerIndex] = {
          type: 'error',
          content: 'error: ' + error.message,
          streaming: false
        }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  const handleClearBrain = async () => {
    setClearStatus('clearing...')
    try {
      const response = await fetch(`${API_URL}/api/clear`, {
        method: 'DELETE'
      })

      const data = await response.json()
      if (data.status === 'success') {
        setClearStatus('cleared')
        setShowClearConfirm(false)
        setMessages([])
        setTimeout(() => setClearStatus(''), 2000)
      } else {
        setClearStatus('error: ' + data.message)
      }
    } catch (error) {
      setClearStatus('error: ' + error.message)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>brain</h1>
        <div className="header-controls">
          <Link to="/inputs" className="nav-link">inputs</Link>
        </div>
      </header>

      <div className="two-column-layout">
        {/* Left Column - Input Data */}
        <div className="column left-column">
          <div className="column-header">
            <h2>input data</h2>
          </div>

          <div className="section">
            <div className="section-title">
              add text
              <span className="live-token-count">{estimateTokenCount(textInput)} tokens</span>
            </div>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value.toLowerCase())}
              placeholder="type anything you want to remember..."
            />
            <button className="button" onClick={handleTextSubmit}>
              store
            </button>
            {textStatus && (
              <div className={`status-message ${textStatus === 'stored' || textStatus.includes('tokens') ? 'success' : textStatus.startsWith('error') ? 'error' : ''}`}>
                {textStatus}
              </div>
            )}
          </div>

          <div className="section">
            <div className="section-title">add file</div>
            <div className="file-input-wrapper">
              <label htmlFor="file-input" className="file-input-label">
                choose file
              </label>
              <input
                id="file-input"
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
              />
              {file && <span className="file-name">{file.name}</span>}
            </div>
            <div></div>
            <button className="button" onClick={handleFileSubmit} disabled={!file}>
              store
            </button>
            {fileStatus && (
              <div className={`status-message ${fileStatus === 'stored' || fileStatus.includes('tokens') ? 'success' : fileStatus.startsWith('error') ? 'error' : ''}`}>
                {fileStatus}
              </div>
            )}
          </div>

          <div className="section">
            <div className="section-title danger">clear brain</div>
            {!showClearConfirm ? (
              <button
                className="button danger"
                onClick={() => setShowClearConfirm(true)}
              >
                clear all data
              </button>
            ) : (
              <div className="clear-confirm">
                <div className="warning-text">
                  this will permanently delete all stored data. are you sure?
                </div>
                <div className="clear-actions">
                  <button className="button danger" onClick={handleClearBrain}>
                    yes, clear everything
                  </button>
                  <button className="button" onClick={() => setShowClearConfirm(false)}>
                    cancel
                  </button>
                </div>
              </div>
            )}
            {clearStatus && (
              <div className={`status-message ${clearStatus === 'cleared' ? 'success' : clearStatus.startsWith('error') ? 'error' : ''}`}>
                {clearStatus}
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Query */}
        <div className="column right-column">
          <div className="column-header">
            <h2>query</h2>
            {(availableProviders.openai_available && availableProviders.ollama_available) && (
              <button
                className="llm-toggle-button"
                onClick={() => setLlmProvider(prev => prev === 'chatgpt' ? 'ollama' : 'chatgpt')}
              >
                {llmProvider === 'chatgpt' ? 'chatgpt' : 'ollama'}
              </button>
            )}
            {!availableProviders.openai_available && availableProviders.ollama_available && (
              <span className="llm-label">ollama</span>
            )}
            {availableProviders.openai_available && !availableProviders.ollama_available && (
              <span className="llm-label">chatgpt</span>
            )}
          </div>

          <div className="section chat-section">
            {messages.length > 0 && (
              <>
                <div className="chat-header">
                  <span className="chat-title">conversation history</span>
                  <button
                    className="button clear-chat-button"
                    onClick={() => setMessages([])}
                    title="Clear chat history"
                  >
                    clear chat
                  </button>
                </div>
                <div className="messages">
                  {messages.map((msg, idx) => (
                    <div key={idx} className="message">
                      <div className="message-label">
                        {msg.type === 'question' ? 'you' : msg.type === 'error' ? 'error' : 'brain'}
                        {msg.streaming && (
                          <span className="streaming-indicator">●</span>
                        )}
                        {msg.tokens && (
                          <span className="token-count" title={`Query: ${msg.tokens.query} | Context: ${msg.tokens.context} | Answer: ${msg.tokens.answer}`}>
                            {msg.tokens.total} tokens
                          </span>
                        )}
                      </div>
                      <div className="message-content">
                        {msg.content}
                        {msg.streaming && <span className="cursor">▋</span>}
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </>
            )}

            <div className="question-input-container">
              <div className="question-input-header">
                <span className="live-token-count">{estimateTokenCount(question)} tokens</span>
              </div>
              <div className="question-input">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value.toLowerCase())}
                  placeholder="ask anything..."
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleAsk()
                    }
                }}
                />
                <button className="button" onClick={handleAsk} disabled={loading}>
                  {loading ? 'thinking...' : 'ask'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
